from konlpy.tag import Komoran
from typing import List, Optional, Tuple
import os

def load_komoran_with_dapa_terms() -> Komoran:
    """
    Creates and returns a Komoran instance with your DAPA terms loaded as proper nouns
    """
    try:
        # File paths
        terms_file = "dapa_exact_terms.txt"
        dict_file = "dapa_dictionary.txt"

        # Check if input file exists
        if not os.path.exists(terms_file):
            print(f"File not found: {terms_file}")
            print("Using basic Komoran without custom dictionary")
            return Komoran()

        # Create dictionary from your terms
        create_dictionary_file(terms_file, dict_file)

        # Create Komoran instance with custom dictionary
        print(f"Loading Komoran with custom dictionary: {dict_file}")
        komoran = Komoran(userdic=dict_file)

        print("✓ Komoran loaded successfully with DAPA terms dictionary")
        return komoran

    except Exception as e:
        print(f"Error loading dictionary: {e}")
        print("Falling back to basic Komoran...")
        return Komoran()


def create_dictionary_file(input_file: str, output_file: str) -> None:
    """
    Creates dictionary file from dapa_exact_terms.txt in KoNLPy format
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        dictionary_entries = set()
        processed_count = 0

        print(f"Reading DAPA terms from: {input_file}")

        for term in lines:
            term = term.strip()
            if term and not term.startswith('#'):
                processed_count += 1

                # Add original term as proper noun
                dictionary_entries.add(f"{term}\tNNP")

                # Add version without spaces for better matching
                no_space_term = term.replace(' ', '')
                if no_space_term != term and len(no_space_term) > 1:
                    dictionary_entries.add(f"{no_space_term}\tNNP")

                # Add version with normalized spaces
                normalized_term = ' '.join(term.split())
                if normalized_term != term:
                    dictionary_entries.add(f"{normalized_term}\tNNP")

                # Show progress for first few terms
                if processed_count <= 3:
                    print(f"  Processing: [{term}]")

        # Write dictionary file in KoNLPy format
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in sorted(dictionary_entries):
                f.write(f"{entry}\n")

        print(f"✓ Processed {processed_count} DAPA terms")
        print(f"✓ Created dictionary with {len(dictionary_entries)} entries")
        print(f"✓ Dictionary saved to: {output_file}")

    except Exception as e:
        print(f"Error creating dictionary file: {e}")
        raise


def analyze_text(text: str, komoran: Optional[Komoran] = None) -> dict:
    """
    Analyze text with DAPA terms dictionary
    """
    if komoran is None:
        komoran = load_komoran_with_dapa_terms()

    try:
        # Get morphological analysis (word, POS tag)
        pos_tags = komoran.pos(text)

        # Get nouns only
        nouns = komoran.nouns(text)

        # Get morphemes only
        morphs = komoran.morphs(text)

        return {
            'pos_tags': pos_tags,  # [(word, POS), ...]
            'nouns': nouns,  # [noun1, noun2, ...]
            'morphs': morphs,  # [morph1, morph2, ...]
            'text': text
        }
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return {'error': str(e), 'text': text}


def extract_proper_nouns(text: str, komoran: Optional[Komoran] = None) -> List[str]:
    """
    Extract only proper nouns (NNP) - these should be your DAPA terms
    """
    if komoran is None:
        komoran = load_komoran_with_dapa_terms()

    try:
        pos_tags = komoran.pos(text)
        proper_nouns = [word for word, pos in pos_tags if pos == 'NNP']
        return proper_nouns
    except Exception as e:
        print(f"Error extracting proper nouns: {e}")
        return []


def extract_dapa_terms(text: str, komoran: Optional[Komoran] = None) -> List[str]:
    """
    Alias for extract_proper_nouns - extracts DAPA terms from text
    """
    return extract_proper_nouns(text, komoran)


class DapaKomoranLoader:
    """
    Simple interface with caching for DAPA terms analysis
    """
    _komoran_instance = None

    @classmethod
    def get_komoran(cls):
        """Returns Komoran instance with DAPA terms loaded (cached)"""
        if cls._komoran_instance is None:
            print("Initializing Komoran with DAPA dictionary...")
            cls._komoran_instance = load_komoran_with_dapa_terms()
        return cls._komoran_instance

    @classmethod
    def analyze(cls, text: str) -> dict:
        """Analyze text and return comprehensive results"""
        komoran = cls.get_komoran()
        return analyze_text(text, komoran)

    @classmethod
    def get_dapa_terms(cls, text: str) -> List[str]:
        """Extract only DAPA terms (proper nouns) from text"""
        komoran = cls.get_komoran()
        return extract_dapa_terms(text, komoran)

    @classmethod
    def get_all_nouns(cls, text: str) -> List[str]:
        """Get all nouns from text"""
        komoran = cls.get_komoran()
        try:
            return komoran.nouns(text)
        except Exception as e:
            print(f"Error getting nouns: {e}")
            return []

    @classmethod
    def get_pos_tags(cls, text: str) -> List[Tuple[str, str]]:
        """Get POS tagged results"""
        komoran = cls.get_komoran()
        try:
            return komoran.pos(text)
        except Exception as e:
            print(f"Error getting POS tags: {e}")
            return []


def test_dapa_dictionary():
    """
    Test the DAPA dictionary with sample texts
    """
    print("=== Testing DAPA Terms Dictionary ===\n")

    # Load Komoran with DAPA terms
    komoran = load_komoran_with_dapa_terms()

    # Test texts with likely DAPA terms
    test_texts = [
        "F-15K 전투기와 HOT 대전차 미사일을 검토한다.",
        "방위사업청에서 88전차 도입을 논의했다.",
        "CN-235 시뮬레이터를 통한 조종사 훈련이 시작됐다.",
        "2단계 경쟁 입찰로 30mm 자주 대공포를 도입한다.",
        "BOA와 BOM 시스템을 통해 CC 인증을 받았다.",
        "[기만기검사기 전원공급] 기만기검사기는 잠수함용 부유식기만기로 점검전원을 공급하여야 한다."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: {text}")

        try:
            # Get POS tags to see all analysis
            pos_tags = komoran.pos(text)
            print(f"  POS Tags: {pos_tags}")

            # Get all nouns
            all_nouns = komoran.nouns(text)
            print(f"  All Nouns: {all_nouns}")

            # Get only DAPA terms (proper nouns)
            dapa_terms = extract_dapa_terms(text, komoran)
            print(f"  DAPA Terms: {dapa_terms}")
            print()

        except Exception as e:
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    test_dapa_dictionary()

# Usage examples:
"""
# Method 1: Direct usage
komoran = load_komoran_with_dapa_terms()
dapa_terms = extract_dapa_terms("F-15K 전투기를 도입한다.", komoran)
print(dapa_terms)  # ['F-15K']

# Method 2: Using the class (recommended for multiple calls)
dapa_terms = DapaKomoranLoader.get_dapa_terms("HOT 대전차 미사일을 검토한다.")
print(dapa_terms)  # ['HOT', '대전차', '미사일'] or ['HOT대전차미사일']

# Method 3: Full analysis
result = DapaKomoranLoader.analyze("CN-235 시뮬레이터로 훈련한다.")
print(result['pos_tags'])  # All POS tagged words
print(result['nouns'])     # All nouns
"""