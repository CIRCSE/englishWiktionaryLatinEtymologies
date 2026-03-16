import gzip
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import List, Dict, NamedTuple, Any
from urllib.parse import quote

import requests
from progiter import ProgIter
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD


"""
Project overview
----------------

This file implements the full pipeline used to extract, transform, curate,
enrich, and serialize etymological data from Wiktionary into RDF using the
lemonEty and OntoLex-Lemon vocabularies.

The pipeline focuses on etymological chains involving Latin and produces
structured linked-data resources describing lexical entries, their
historical ancestors, and the etymological links connecting them.

The resulting dataset is published as part of the LiLa Linked Data
infrastructure and can be explored here:

https://lila-erc.eu/data/lexicalResources/englishWiktionaryLatinEtymologies/Lexicon

Pipeline overview
-----------------

1. Utilities
2. Data extraction
3. Data transformation
4. Data curation
5. Linking utilities
6. RDF serialization


The pipeline is intentionally modular so that its components can be easily
modified or replaced. This makes it straightforward to adapt the system to
extract etymological chains for other languages. Some components can also be
repurposed to extract different types of structured information from
Wiktionary. 

Furthermore, another important design principle of this pipeline is the use of
precomputed intermediate datasets. Intermediate stages are written to disk
and reloaded when needed rather than recomputed each time the pipeline runs.

This greatly improves performance and reproducibility. For example,
etymology chains extracted from Wiktionary templates are saved to a file and
later reloaded for enrichment and RDF serialization, avoiding the need to
re-run the expensive extraction stage repeatedly.
"""


# -------------- Utilities --------------

def timer(func):
    """
    Decorator that measures and prints the execution time of the decorated function.

    The wrapped function is executed normally, but its runtime is measured using
    time.perf_counter() and printed to stdout after completion.

    Parameters
    ----------
    func : Callable
       The function whose execution time will be measured.

    Returns
    -------
    Callable
       A wrapped version of the input function that prints its execution time
       when called.

    Notes
    -----
    This decorator is intended for lightweight performance monitoring during
    pipeline execution and debugging. It does not modify the return value or
    behavior of the decorated function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# -------------- Data extraction --------------


def parse_json_line(line: str, line_no: int, warn: bool = False):
    """
    Parse a single JSON line from the Wiktionary dump.

    Attempts to deserialize a JSON object from the provided line. Empty
    lines are ignored. If the line contains malformed JSON, the function
    optionally emits a warning and returns None.

    Parameters
    ----------
    line : str
        A single line from the JSONL Wiktionary dump.
    line_no : int
        Line number in the source file, used for reporting malformed lines.
    warn : bool, optional
        If True, print a warning to stderr when malformed JSON is encountered.

    Returns
    -------
    dict | None
        The parsed JSON object if successful, otherwise None.
    """
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        if warn:
            print(f"[warn] skip malformed line {line_no}", file=sys.stderr)

        return None


def count_lines_in_gz(gz_source_file: str) -> int:
    """
    Count the number of lines in a gzipped JSONL file.

    This function is primarily used to determine the total number of
    records in the Wiktionary dump so that progress bars can display
    accurate percentages during processing.

    Parameters
    ----------
    gz_source_file : str
        Path to the gzipped JSONL file.

    Returns
    -------
    int
        Total number of lines contained in the file.
    """
    with gzip.open(gz_source_file, "rt", encoding="utf-8", errors="strict") as f:
        return sum(1 for _ in f)


# List of allowed languages so we can't just pass any string to the
# get_languages_templates_from_gz function. The choice of languages
# is arbitrary. Modify the list however you like.
allowed_languages = ["English", "Latin", "Spanish", "Italian", "German"]


@timer
def get_language_templates_from_gz(gz_source_file: str, target_lang: str, target_file: str = None):
    """
    Extract etymology template data for a specific language from a Wiktionary dump.

    The function scans a gzipped JSONL Wiktionary export and collects the
    `etymology_templates` field for entries belonging to the specified
    language. The extracted templates are grouped by lemma and optionally
    written to a JSON file.

    Parameters
    ----------
    gz_source_file : str
        Path to the gzipped Wiktionary JSONL dump.
    target_lang : str
        Language name to filter entries by (must be present in
        `allowed_languages`).
    target_file : str | None, optional
        Output file where the extracted templates will be saved as JSON.
        If None, a timestamped filename is automatically generated.

    Returns
    -------
    dict[str, list]
        Mapping from lemma to a list of etymology template structures
        extracted from the dump.

    Notes
    -----
    This step constitutes the **data extraction stage** of the pipeline.
    It isolates the raw etymology template structures that will later be
    transformed into etymological chains.
    """
    if target_lang not in allowed_languages:
        raise ValueError(
            f"Language {target_lang} is not allowed. The following languages are allowed: {allowed_languages}")
    if target_file is None:
        target_file = f'{target_lang.lower()}_etym_templates_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    final_etymology_templates = defaultdict(list)
    print("Computing file length to track progress better... this might take a while for big files.")
    total_file_length = count_lines_in_gz(gz_source_file)
    with gzip.open(gz_source_file, "rt", encoding="utf-8", errors="strict") as file_input:
        for line_index, line in ProgIter(
                enumerate(file_input, start=1),
                total=total_file_length,
                desc="Scanning Wiktionary",
                freq=1.0,
                show_percent=True
        ):
            wiktionary_entry = parse_json_line(line, line_index)
            if not wiktionary_entry:
                continue
            if not (wiktionary_entry.get("lang") == target_lang):
                continue
            etymology_templates = wiktionary_entry.get("etymology_templates")
            word = wiktionary_entry.get("word")
            if not etymology_templates:
                final_etymology_templates[word].append({})
            else:
                final_etymology_templates[word].append(etymology_templates)
    with open(target_file, "w") as file:
        json.dump(final_etymology_templates, file)
    return final_etymology_templates


# -------------- Data transformation --------------


# A nice abstraction layer so the code is more readable. Can perfectly be removed by
# replacing .form with [0] and .lang with [1], an initializing regular tuples instead
# of initializing AncestryNode objects
class AncestryNode(NamedTuple):
    """
    Lightweight representation of one node in an etymological chain.

    Each node stores a lexical form and the Wiktionary language code
    associated with that form. It is used as the normalized in-memory
    representation of ancestry information extracted from template objects.

    Attributes
    ----------
    form : str
        Lexical form, lemma, or reconstructed form extracted from the template.
    lang : str
        Wiktionary language code corresponding to the form.
    """
    form: str
    lang: str


# If Wiktionary etymologies for a language do not follow this pattern, the pipeline won't work.
# When implementing etymology chain building for another language, please check whether its
# etymology templates are included here. If not, extend the list.
RELEVANT = {"inh", "inh+", "der", "der+", "bor", "bor+", "root"}
INH_LIKE = {"inh", "inh+", "der", "der+", "bor", "bor+", "ubor", "slbor"}  # unadapted/semi-learned borrowings
INH_LIKE_LITE = {"inh-lite", "der-lite", "bor-lite"}  # seen variants; tolerate if present
ROOT_LIKE = {"root"}
SKIP_NAMES = {
    "der-lite/lang", "m-lite", "af", "affix", "nonlemma", "glossary",
    "etymid", "etyl", "gentrade", "dbt", "doublet", "dercat", "intnat",
    "etydate", "univ", "etymon"  # by default: skip; it's too free-form
}


def get_node_from_template_object(etymology_template: dict) -> AncestryNode | None:
    """
    Convert a single Wiktionary etymology template object into an ancestry node.

    The function inspects the template name and arguments and, when the
    template belongs to a supported etymological pattern, extracts the
    source form and source language code as an `AncestryNode`. Templates
    that are irrelevant, unsupported, or too free-form for reliable
    parsing are ignored.

    Parameters
    ----------
    etymology_template : dict
        A single template object from the `etymology_templates` field of a
        Wiktionary entry.

    Returns
    -------
    AncestryNode | None
        An `AncestryNode` containing the extracted form and language code
        if the template matches a supported pattern, otherwise None.

    Notes
    -----
    This function assumes that relevant etymological templates encode the
    source language in argument `"2"` and the source form in argument `"3"`,
    which is the pattern followed by the supported Wiktionary templates.
    """
    name = etymology_template.get("name", None)
    if not name or name in SKIP_NAMES:
        return None

    # Get the kind of etymology this template object describes
    args = etymology_template.get("args", {})

    # Standard ancestry
    if name in INH_LIKE:
        # args["2"] = source language code, args["3"] = lemma/form
        lang = args.get("2", "")
        lemma = args.get("3", "")
        return AncestryNode(lemma, lang)

    # Lite ancestry (same arg positions as standard)
    if name in INH_LIKE_LITE:
        lang = args.get("2", "")
        lemma = args.get("3", "")
        return AncestryNode(lemma, lang)

    # Roots
    if name in ROOT_LIKE:
        root_lang = args.get("2", "")
        root_form = args.get("3", "")
        return AncestryNode(root_form, root_lang)

    return None


def build_etymology_chain(templates: List[dict]) -> List[AncestryNode]:
    """
    Build an etymological chain from a list of Wiktionary template objects.

    The function processes each template object in sequence, converts
    supported templates into `AncestryNode` objects, and returns the
    resulting list of ancestry nodes.

    Parameters
    ----------
    templates : list[dict]
        List of template objects associated with a single Wiktionary
        etymology section.

    Returns
    -------
    list[AncestryNode]
        Ordered list of ancestry nodes extracted from the input templates.
        Templates that do not map to a supported ancestry relation are skipped.

    Notes
    -----
    The output order reflects the order in which relevant templates appear
    in the source data. A later step may reverse this order to obtain the
    desired historical direction of the chain.
    """
    if not templates:
        return []

    steps: List[AncestryNode] = []
    for template_object in templates:
        node: AncestryNode | None = get_node_from_template_object(template_object)
        if node:
            steps.append(node)
    return steps


def reverse_etymology_chain(etymology_chain: List[AncestryNode]) -> List[AncestryNode]:
    """
    Reverse the order of an etymological chain.

    This helper is used to convert the chain into the desired historical
    direction for downstream processing and RDF serialization.

    Parameters
    ----------
    etymology_chain : list[AncestryNode]
        Etymological chain represented as a list of ancestry nodes.

    Returns
    -------
    list[AncestryNode]
        Reversed version of the input chain.
    """
    return list(reversed(etymology_chain))


@timer
def transform_language_templates_into_chains(etymology_templates: Dict[str, Any], target_lang: str,
                                             target_file: str = None) -> Dict[str, Any]:
    """
    Transform extracted etymology templates into normalized etymological chains.

    For each word in the extracted template index, the function converts each
    template list into a chain of `AncestryNode` objects, reverses the chain
    into the desired historical order, and stores all non-empty chains in a
    new index keyed by word. The resulting structure can optionally be written
    to a JSON file.

    Parameters
    ----------
    etymology_templates : dict[str, Any]
        Mapping from word to one or more raw etymology template lists
        extracted from the Wiktionary dump.
    target_lang : str
        Language whose template data is being transformed. Must be included
        in `allowed_languages`.
    target_file : str | None, optional
        Output JSON file where the chain index will be saved. If None,
        a timestamped filename is generated automatically.

    Returns
    -------
    dict[str, Any]
        Mapping from word to a list of normalized etymological chains.

    Raises
    ------
    ValueError
        If `target_lang` is not included in `allowed_languages`.

    Notes
    -----
    This function constitutes the **data transformation stage** of the
    pipeline. It converts raw template structures into a cleaner and more
    semantically meaningful representation suitable for curation, enrichment,
    and RDF generation.
    """
    chain_index = {}
    if target_lang not in allowed_languages:
        raise ValueError(
            f"Language {target_lang} is not allowed. The following languages are allowed: {allowed_languages}")
    if target_file is None:
        target_file = f'{target_lang.lower()}_etym_chains_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    for word, templates_list in ProgIter(
            etymology_templates.items(),
            total=len(etymology_templates),
            desc="Building etymology chains",
            show_percent=True
    ):
        chains = []

        for templates in templates_list:
            chain = reverse_etymology_chain(build_etymology_chain(templates))
            if chain:
                chains.append(chain)

        if chains:
            chain_index[word] = chains
    with open(target_file, "w") as file:
        json.dump(chain_index, file)
    return chain_index


def load_chain_index(path: str) -> dict:
    """
     Load a serialized chain index and reconstruct `AncestryNode` objects.

     This function reads a JSON file containing etymological chains produced
     during the transformation stage and converts the serialized node
     representations back into `AncestryNode` instances for in-memory use.

     Parameters
     ----------
     path : str
         Path to the JSON file containing the serialized chain index.

     Returns
     -------
     dict
         Mapping from word to a list of etymological chains, where each chain
         is a list of `AncestryNode` objects.

     Notes
     -----
     Chains are stored in JSON as simple lists of `[form, lang]` pairs.
     This function restores the richer in-memory representation used by
     the pipeline by converting those pairs into `AncestryNode` objects.

     This step effectively *hydrates* the serialized chain data back into
     the structured form expected by downstream enrichment and RDF
     serialization functions.
     """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        word: [
            [AncestryNode(form=node[0], lang=node[1]) for node in chain]
            for chain in chains
        ]
        for word, chains in raw.items()
    }

# -------------- Data curation --------------


def detect_imperfect_chains(etymology_chains: dict):
    """
    Classify etymological chains according to simple structural quality checks.

    The function inspects each chain and assigns it to one of several
    categories reflecting potential imperfections, such as empty forms,
    repeated language codes, inline annotations, morpheme-like forms,
    enumerations, or leftover markup characters. Chains that do not trigger
    any of these conditions are classified as valid.

    Parameters
    ----------
    etymology_chains : dict
        Mapping from word to a list of etymological chains. Each chain is
        expected to be an ordered sequence of items containing a form and a
        language code.

    Returns
    -------
    list[dict]
        List of classification buckets. Each bucket is a dictionary with:
        - `"class"`: the name of the category
        - `"chains"`: a mapping from word to the list of chains assigned to
          that category

    Notes
    -----
    This function constitutes the **data curation stage** of the pipeline.
    Its purpose is to separate apparently valid chains from chains that may
    require manual inspection, filtering, or later refinement before
    enrichment and RDF serialization.

    The current quality checks are heuristic and intentionally simple. A chain
    is assigned to the first matching non-valid category encountered during
    inspection.

    WARNING: these chains are not AncestryNode objects arrays, so we access form
    by doing [0] and lang by doing [1]. This was done this way to load an existing
    chain index from a file, divide it into buckets and save them into new JSON files.
    Among these new JSON files, we'd have the valid chains JSON file.

    By classifying the invalid chains, it's easier for us to review them to look for
    false negatives.

    """
    empty = {"class": "empty", "chains": defaultdict(list)}
    repeated_lang = {"class": "repeated_lang", "chains": defaultdict(list)}
    annotated = {"class": "annotated", "chains": defaultdict(list)}
    morpheme = {"class": "morpheme", "chains": defaultdict(list)}
    enumeration = {"class": "enumeration", "chains": defaultdict(list)}
    markup = {"class": "markup", "chains": defaultdict(list)}
    valid = {"class": "valid", "chains": defaultdict(list)}
    classification = [valid, empty, repeated_lang, annotated, morpheme, enumeration, markup]
    for word, chains in etymology_chains.items():
        for chain in chains:
            langs = []
            found = False
            for item in chain:
                form = item[0].strip()
                lang = item[1]
                if lang in langs:
                    repeated_lang["chains"][word].append(chain)
                    found = True
                    break
                langs.append(lang)
                if form == "" or form == "-":
                    empty["chains"][word].append(chain)
                    found = True
                    break
                if "(" in form:
                    annotated["chains"][word].append(chain)
                    found = True
                    break
                if form.startswith("-"):
                    morpheme["chains"][word].append(chain)
                    found = True
                    break
                if "," in form:
                    enumeration["chains"][word].append(chain)
                    found = True
                    break
                if "<" in form or ">" in form or "[" in form or "]" in form or "{" in form or "}" in form:
                    markup["chains"][word].append(chain)
                    found = True
                    break
            if found:
                print("NOT VALID", word, chain)
            else:
                valid["chains"][word].append(chain)
                print("VALID", word, chain)

    # Convert defaultdicts to normal dicts
    for bucket in classification:
        bucket["chains"] = dict(bucket["chains"])
    return classification


# -------------- Linking utilities (for the enrichment step) --------------
# WARNING: when linking to external sources, bear in mind that is someone's server!
# Please be considerate, especially if you're operating on a big scale!

# Function to link to Wiktionary

# Reuse a session for efficiency
_SESSION = requests.Session()

# Set HEADERS["User-Agent"] to identify yourself, e.g. name/project + contact email.
# Example: {"User-Agent": "EtymologyRDFBot/0.1 (you@example.com) requests/2.32.4"}
# This matters because Wiktionary admins can contact you if your script causes problems.
# A clear User-Agent is good net etiquette and helps distinguish responsible research traffic from abuse.
# Never fake browser headers to hide your script; be transparent and keep request volume low.

# WARNING: this header is just an example! It will NOT work!
HEADERS = {'User-Agent': 'YourBotName/0.1 (yourEmail@gmail.com) Python/3.11 requests/2.32.4 (or whatever youre using)'}

PROTO_LANG_CANONICAL = {
    "ine-pro": "Proto-Indo-European",
    "gem-pro": "Proto-Germanic",
    "gmw-pro": "Proto-West_Germanic",
    "itc-pro": "Proto-Italic",
    "cel-pro": "Proto-Celtic",
    "sla-pro": "Proto-Slavic",
    "grk-pro": "Proto-Hellenic",
}


def find_wiktionary_entry_for_word(word: str,
                                   lang_code: str | None = None,
                                   timeout: float = 5.0,
                                   request_delay: float = 1.0) -> str | None:
    """
    Resolve a lexical form to a Wiktionary entry URL when possible.

    The function generates one or more candidate English Wiktionary URLs for
    the given form and checks whether any of them exist. For reconstructed
    proto-language forms, it also attempts the corresponding
    `Reconstruction:Language/Form` page when the language code can be mapped
    to a canonical proto-language name.

    Parameters
    ----------
    word : str
        Lexical form to look up. The form may include a leading asterisk,
        as is common for reconstructed proto-forms.
    lang_code : str | None, optional
        Wiktionary language code associated with the form. If the code
        corresponds to a supported proto-language, an additional
        reconstruction-page candidate is generated.
    timeout : float, optional
        Maximum number of seconds to wait for each HTTP request.
    request_delay : float, optional
        Number of seconds to wait before each outgoing request, used to avoid
        stressing the remote server.

    Returns
    -------
    str | None
        A Wiktionary URL if a candidate page responds successfully with HTTP
        status 200, otherwise None.

    Notes
    -----
    This function is part of the **enrichment step** of the pipeline. It is
    designed to be conservative and server-friendly: requests are rate-limited,
    a descriptive User-Agent should be supplied, and failed requests are
    treated as unresolved links rather than fatal errors.
    """
    lemma = word.strip()
    proto_lemma = word.lstrip("*").strip()

    candidates: List[str] = ["https://en.wiktionary.org/wiki/" + quote(lemma, safe="")]

    if proto_lemma and proto_lemma != lemma:
        candidates.append("https://en.wiktionary.org/wiki/" + quote(proto_lemma, safe=""))

    if lang_code and lang_code.endswith("-pro") and proto_lemma:
        lang_name = PROTO_LANG_CANONICAL.get(lang_code)
        if lang_name:
            title = f"Reconstruction:{lang_name}/{proto_lemma}"
            candidates.append("https://en.wiktionary.org/wiki/" + quote(title, safe=":/"))

    for url in candidates:
        time.sleep(request_delay)
        try:
            resp = _SESSION.head(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 405:  # method not allowed
                resp = _SESSION.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            return url
    return None


# Functions for linking to LiLa: https://lila-erc.eu/#about-1


LILA_BASE = "https://lila-erc.eu/data/id/"
LINK_RE = re.compile(r"^lilaLemma:(\d+)$")
LILA_URL = "https://lila-erc.eu/LiLaTextLinker/processText"


def query_lila(word: str) -> dict:
    """
    Send a lexical form to the LiLa Text Linker service and return the raw response.

    The function submits the given word to the LiLa endpoint as JSON and
    returns the decoded JSON response produced by the service.

    Parameters
    ----------
    word : str
        Lexical form to be linked against LiLa.

    Returns
    -------
    dict
        Parsed JSON response returned by the LiLa Text Linker service.

    Raises
    ------
    requests.HTTPError
        If the remote service responds with an HTTP error status.

    Notes
    -----
    This function performs no interpretation of the returned payload. It
    simply exposes the raw service response for downstream processing.
    """
    payload = {"text": word}
    response = requests.post(LILA_URL, json=payload, timeout=10)
    response.raise_for_status()  # raises if not 200
    return response.json()


def link_lemma_to_lila(lemma: str) -> str | None:
    """
     Attempt to link a lemma to a LiLa lemma identifier.

     The function queries the LiLa Text Linker service and extracts the first
     available linking key from the returned response structure. If no valid
     linking key can be extracted, the function returns None.

     Parameters
     ----------
     lemma : str
         Lemma to be linked to LiLa.

     Returns
     -------
     str | None
         LiLa linking key such as ``"lilaLemma:103739"`` if linking succeeds,
         otherwise None.

     Notes
     -----
     This function is intentionally forgiving: linking failures, unexpected
     response structures, or missing data do not interrupt pipeline execution.
     Instead, unresolved lemmas are left unlinked so the enrichment step can
     proceed.
     """
    try:
        link_to_lila = query_lila(lemma)
        # In cases where multiple candidate lemmas are returned (e.g., due to homography),
        # the pipeline selects the first candidate provided by the service, which corresponds to
        # the top-ranked linking result.
        linking_key = link_to_lila["sentences"][0][0]["linking"][0]
        print(f"Linked {lemma} to LiLa: {linking_key}")
        return linking_key
    except Exception as e:
        return None


def create_lemma_uri(linking_key: str) -> str:
    """
    Convert a LiLa linking key into a full LiLa resource URI.

    Supported linking keys currently include lemma identifiers and
    hypolemma identifiers. The function validates the key format and
    returns the corresponding full URI string.

    Parameters
    ----------
    linking_key : str
        LiLa linking key, for example ``"lilaLemma:103739"`` or
        ``"lilaIpoLemma:12345"``.

    Returns
    -------
    str
        Full LiLa URI corresponding to the linking key.

    Raises
    ------
    ValueError
        If the linking key has an invalid format or an unsupported prefix.

    Notes
    -----
    This helper converts compact identifiers returned by the LiLa linking
    service into dereferenceable URI strings suitable for RDF serialization.
    """
    key = linking_key.strip()
    prefix, lemma_id = key.split(":", 1)
    if not lemma_id.isdigit():
        raise ValueError(f"Invalid linking key format: {linking_key!r}")
    if prefix == "lilaLemma":
        return f"{LILA_BASE}lemma/{lemma_id}"
    if prefix == "lilaIpoLemma":
        return f"{LILA_BASE}hypolemma/{lemma_id}"
    raise ValueError(f"Unknown LiLa linking key prefix: {prefix!r}")


# -------------- RDF Serialization --------------

def enrich_chain(etymology_chain: List[AncestryNode], wiktionary_url: bool = True, lila_linking: bool = True) -> List[
    Dict[str, object]]:
    """
     Enrich an etymological chain with external linking information.

     The function converts an in-memory chain of `AncestryNode` objects into
     a list of dictionaries and optionally augments each node with a
     Wiktionary URL and, for Latin nodes, a LiLa linking key.

     Parameters
     ----------
     etymology_chain : list[AncestryNode]
         Etymological chain to enrich.
     wiktionary_url : bool, optional
         If True, attempt to resolve each node to a Wiktionary entry URL.
     lila_linking : bool, optional
         If True, attempt to link Latin nodes to LiLa.

     Returns
     -------
     list[dict[str, object]]
         List of enriched node dictionaries. Each dictionary contains at least
         `"form"` and `"lang"`, and may additionally contain `"url"` and
         `"lila_link"`.

     Notes
     -----
     This function bridges the transformation and serialization stages of the
     pipeline by converting normalized ancestry nodes into a richer structure
     suitable for RDF generation.
     """
    enriched_chain = []
    for node in etymology_chain:
        enriched_chain_node = {"form": node.form, "lang": node.lang}
        if wiktionary_url:
            url = find_wiktionary_entry_for_word(node.form, node.lang)
            enriched_chain_node["url"] = url
        if lila_linking and node.lang == "la":
            lila_lemma = link_lemma_to_lila(node.form)
            enriched_chain_node["lila_link"] = lila_lemma
        enriched_chain.append(enriched_chain_node)
    return enriched_chain


LEMONETY = Namespace(
    "http://lari-datasets.ilc.cnr.it/lemonEty#")  # lemonEty vocabulary :contentReference[oaicite:3]{index=3}
ONTOLEX = Namespace("http://www.w3.org/ns/lemon/ontolex#")  # OntoLex-Lemon :contentReference[oaicite:4]{index=4}
SCHEMA = Namespace("http://schema.org/")  # schema.org :contentReference[oaicite:5]{index=5}
OWL = Namespace("http://www.w3.org/2002/07/owl#")


def wiktionary_iso_639_to_bcp47(lang_code: str) -> str:
    """
     Map a Wiktionary language code to a language tag suitable for RDF literals.

     Standard alphabetic language codes of length two or three are returned
     in lowercase. Non-standard or reconstructed language codes are converted
     into private-use tags of the form ``x-...``.

     Parameters
     ----------
     lang_code : str
         Wiktionary language code.

     Returns
     -------
     str
         Language tag to be used in RDF literal serialization.

     Notes
     -----
     This is a pragmatic mapping used for RDF generation. Reconstructed or
     otherwise non-standard Wiktionary codes are represented as private-use
     tags rather than strict standard language tags.
     """
    code = lang_code.strip()
    if 2 <= len(code) <= 3 and code.isalpha():
        return code.lower()
    # Proto / non-standard: make a private-use tag like "x-ine-pro"
    return f"x-{code}"


def build_lemonety_for_chain(chain: list[dict], base_uri: str, graph: Graph = None) -> Graph:
    """
    Build an RDF graph representing one etymological chain in lemonEty/OntoLex.

    The function converts an enriched etymological chain into RDF resources
    describing:
    - the etymology as a whole
    - the most recent form as an OntoLex lexical entry
    - earlier stages as lemonEty etymons
    - etymological links connecting successive stages

    Parameters
    ----------
    chain : list[dict]
        Enriched etymological chain. Each node is expected to contain at
        least `"form"` and `"lang"`, and may also include `"url"` and
        `"lila_link"`.
    base_uri : str
        Base URI used to mint RDF resource identifiers for the generated
        etymology, lexical entry, etymons, and etymological links.
    graph : rdflib.Graph | None, optional
        Existing RDF graph to which triples will be added. If None, a new
        graph is created and namespace prefixes are bound automatically.

    Returns
    -------
    rdflib.Graph
        RDF graph containing the triples generated for the chain.

    Notes
    -----
    The last node in the chain is treated as the most recent stage and is
    modeled as the main lexical entry. All preceding nodes are modeled as
    etymons. Successive stages are connected by a sequence of
    `lemonEty:EtyLink` resources.

    When available, Wiktionary URLs are attached with `schema:url`, and LiLa
    links are converted into canonical-form URIs. When no LiLa link is
    available, a blank node with an `ontolex:writtenRep` literal is used
    instead.
    """
    if graph is None:
        graph = Graph()
        graph.bind("lemonEty", LEMONETY)
        graph.bind("ontolex", ONTOLEX)
        graph.bind("schema", SCHEMA)

    # Most recent form (last in chain) is the lexical entry
    latest = chain[-1]
    latest_form = latest.get("form")
    latest_lang = latest.get("lang")
    latest_url = latest.get("url")
    latest_lila_link = latest.get("lila_link")
    entry_id = f"{latest_form}-{latest_lang}"
    safe_entry_id = quote(entry_id, safe="")
    # --- Etymology node ---------------------------------------------------
    ety_uri = URIRef(f"{base_uri.rstrip('/')}/etymology/{safe_entry_id}")
    graph.add((ety_uri, RDF.type, LEMONETY.Etymology))
    graph.add((ety_uri, RDFS.label, Literal(f"Etymology of: {latest_form}", datatype=XSD.string)))

    # --- LexicalEntry for the base word (latest stage) ---------------------
    entry_uri = URIRef(f"{base_uri.rstrip('/')}/lex/{safe_entry_id}")
    graph.add((entry_uri, RDF.type, ONTOLEX.LexicalEntry))
    graph.add((entry_uri, RDFS.label, Literal(latest_form, datatype=XSD.string)))
    entry_lang_bcp47 = wiktionary_iso_639_to_bcp47(latest_lang)

    # Link lexical entry to this Etymology
    graph.add((entry_uri, LEMONETY.etymology, ety_uri))
    graph_main_lemma = entry_id.split("-")[0]
    if latest_lila_link:
        graph.add((entry_uri, ONTOLEX.canonicalForm, URIRef(create_lemma_uri(latest_lila_link))))
    else:
        # print(f"Did not find a link to LiLa for {graph_main_lemma}")
        entry_cf = BNode()
        graph.add((entry_uri, ONTOLEX.canonicalForm, entry_cf))
        graph.add((entry_cf, ONTOLEX.writtenRep, Literal(latest_form, lang=entry_lang_bcp47)))

    entry_lang_bcp47 = wiktionary_iso_639_to_bcp47(latest_lang)
    graph.add((entry_uri, SCHEMA.inLanguage, Literal(entry_lang_bcp47, datatype=XSD.language)))
    if latest_url:
        graph.add((entry_uri, SCHEMA.url, URIRef(latest_url)))

    # --- Etymon nodes for all stages --------------------------------------
    etymon_uris: List[URIRef] = []
    etymon_list = chain[:-1]
    # print("Etymons")
    for idx, node in enumerate(etymon_list):
        # print(idx, "->", node)
        form = node.get("form")
        lang = node.get("lang")
        url = node.get("url")
        etymon_lila_link = node.get("lila_link")

        etymon_uri = URIRef(f"{base_uri.rstrip('/')}/etymon/{safe_entry_id}/{len(etymon_list) - idx - 1}")
        etymon_uris.append(etymon_uri)

        # Type & label
        graph.add((etymon_uri, RDF.type, LEMONETY.Etymon))
        graph.add((etymon_uri, RDFS.label, Literal(form, datatype=XSD.string)))

        lang_bcp47_code = wiktionary_iso_639_to_bcp47(lang)
        lila_lemma_uri = None
        if lang == "la":
            if etymon_lila_link:
                lila_lemma_uri = URIRef(create_lemma_uri(etymon_lila_link))
                graph.add((etymon_uri, ONTOLEX.canonicalForm, lila_lemma_uri))
            else:
                cf_node = BNode()
                graph.add((etymon_uri, ONTOLEX.canonicalForm, cf_node))
                graph.add((cf_node, ONTOLEX.writtenRep, Literal(form, lang=lang_bcp47_code)))

        # schema:inLanguage as an xsd:language literal
        graph.add((etymon_uri, SCHEMA.inLanguage, Literal(lang_bcp47_code, datatype=XSD.language)))

        # schema:url if we have a real URL
        if url:
            graph.add((etymon_uri, SCHEMA.url, URIRef(url)))

        # Link etymon to the Etymology hypothesis
        graph.add((ety_uri, LEMONETY.etymon, etymon_uri))

    # --- Etymology links (sequence of EtyLink) -----------------------------
    etylink_uris: list[URIRef] = []
    sequence: list[URIRef] = etymon_uris + [entry_uri]
    for i in range(len(sequence) - 1):
        link_uri = URIRef(f"{base_uri.rstrip('/')}/etylink/{safe_entry_id}/{i + 1}")
        etylink_uris.append(link_uri)
        graph.add((link_uri, RDF.type, LEMONETY.EtyLink))
        graph.add((link_uri, RDFS.label, Literal("Etymology Link", datatype=XSD.string)))

        # Source = earlier stage, Target = later stage
        graph.add((link_uri, LEMONETY.etySource, sequence[i]))
        graph.add((link_uri, LEMONETY.etyTarget, sequence[i + 1]))

        # Attach the link to the Etymology
        graph.add((ety_uri, LEMONETY.hasEtyLink, link_uri))

    # startingLink = first link in the chain
    if etylink_uris:
        graph.add((ety_uri, LEMONETY.startingLink, etylink_uris[0]))

    return graph


def generate_triples_from_chain_index(chain_index: dict, lang: str, nt_filepath: str, limit: int | None = None) -> None:
    """
    Generate N-Triples output from an index of etymological chains.

    For each word in the input chain index, the function appends the target
    word itself as the most recent stage, enriches the resulting chain with
    external links, converts it into an RDF graph, and serializes the triples
    to an N-Triples file.

    Parameters
    ----------
    chain_index : dict
        Mapping from word to one or more etymological chains.
    lang : str
        Language code of the target words being serialized as the most recent
        stage in each chain.
    nt_filepath : str
        Output file path for the serialized N-Triples data.
    limit : int | None, optional
        Maximum number of words to process. If None, all words in the index
        are processed.

    Returns
    -------
    None
        This function writes RDF triples to disk and does not return a value.

    Notes
    -----
    This function constitutes the final **serialization stage** of the
    pipeline. It turns curated and enriched etymological chains into a
    persistent RDF dataset in N-Triples format.
    """
    count = 0
    with open(nt_filepath, "wb") as out:
        for word, chains in chain_index.items():
            print(f"Word {word} has chains:")
            for chain in chains:
                chain_full = chain + [AncestryNode(word, lang)]
                print(chain_full)
                enriched_chain = enrich_chain(chain_full)
                graph = build_lemonety_for_chain(enriched_chain, "http://testingtripleproduction")
                graph.serialize(destination=out, format="nt")
            count += 1
            if limit is not None and count >= limit:
                break


if __name__ == "__main__":
    """
     Example usage of the pipeline.

     This block demonstrates how to load a previously generated chain index
     and serialize it into RDF triples. It is intentionally disabled to
     prevent accidental execution, since the pipeline may perform network
     requests (e.g., Wiktionary or LiLa linking).
     """
    raise Exception("Are you sure you want to run this? This is just a code example")
    latin_chains_file = "clean_latin_chains.json"
    latin_chain_index = load_chain_index(latin_chains_file)
    triples_path = "englishWiktionaryLatinEtymologies_v2.nt"
    generate_triples_from_chain_index(latin_chain_index, "la", triples_path)
