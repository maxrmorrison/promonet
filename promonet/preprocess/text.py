import multiprocessing as mp
import re

import torch
import tqdm

import phonemizer
import unidecode


###############################################################################
# Interface
###############################################################################


def from_string(text):
    """Convert text string to sequence of integer indices"""
    indices = [symbol_to_id(symbol) for symbol in clean_text(text)]
    return torch.tensor(indices, dtype=torch.long)


def from_file(text_file):
    """Convert text on disk to sequence of integer indices"""
    with open(text_file) as file:
        return from_string(file.read())


def from_file_to_file(text_file, output_file):
    """Convert text on disk to sequence of integer indices and save to disk"""
    torch.save(from_file(text_file), output_file)


def from_files_to_files(text_files, output_files):
    """Convert text files to sequences of integer indices and save to disk"""
    with mp.get_context('spawn').Pool() as pool:
        pool.starmap(from_file_to_file, zip(text_files, output_files))
    # iterator = tqdm.tqdm(
    #     zip(text_files, output_files),
    #     total=len(text_files),
    #     desc='Text preprocessing',
    #     dynamic_ncols=True)
    # for text_file, output_file in iterator:
    #     from_file_to_file(text_file, output_file)


###############################################################################
# Text preprocessing
###############################################################################


def clean_text(text):
    """Pipeline for cleaning english text"""
    text = unidecode.unidecode(text)
    text = text.lower()
    text = expand_abbreviations(text)
    phonemes = phonemizer.phonemize(
        text,
        language='en-us',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=True)
    return collapse_whitespace(phonemes)


###############################################################################
# Symbols
###############################################################################


def id_to_symbol(id):
    """Integer to symbol conversion"""
    if not hasattr(id_to_symbol, 'map'):
        id_to_symbol.map = {i: s for i, s in enumerate(symbols)}
    return id_to_symbol.map[id]


def symbol_to_id(symbol):
    """Symbol to integer conversion"""
    if not hasattr(symbol_to_id, 'map'):
        symbol_to_id.map = {s: i for i, s in enumerate(symbols())}
    return symbol_to_id.map[symbol]


def symbols():
    """Symbol cache"""
    if not hasattr(symbols, 'symbols'):
        pad = '_'
        punctuation = ';:,.!?¡¿—…"«»“” '
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        symbols.symbols = list(pad + punctuation + letters + letters_ipa)
    return symbols.symbols


###############################################################################
# Cleaners
###############################################################################


def expand_abbreviations(text):
    """Replaces abbreviated text"""
    # Cache abbreviations
    if not hasattr(expand_abbreviations, 'abbreviations'):
        expand_abbreviations.abbreviations = [
            (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
                ('mrs', 'misess'),
                ('mr', 'mister'),
                ('dr', 'doctor'),
                ('st', 'saint'),
                ('co', 'company'),
                ('jr', 'junior'),
                ('maj', 'major'),
                ('gen', 'general'),
                ('drs', 'doctors'),
                ('rev', 'reverend'),
                ('lt', 'lieutenant'),
                ('hon', 'honorable'),
                ('sgt', 'sergeant'),
                ('capt', 'captain'),
                ('esq', 'esquire'),
                ('ltd', 'limited'),
                ('col', 'colonel'),
                ('ft', 'fort')]]

    # Expand abbreviations
    for regex, replacement in expand_abbreviations.abbreviations:
        text = re.sub(regex, replacement, text)

    return text


def collapse_whitespace(text):
    """Removes unnecessary whitespace"""
    if not hasattr(collapse_whitespace, 'regex'):
        collapse_whitespace.regex = re.compile(r'\s+')
    return re.sub(collapse_whitespace.regex, ' ', text)
