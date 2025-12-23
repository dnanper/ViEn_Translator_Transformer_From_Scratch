"""
Test and Use Trained Tokenizer
Load tokenizer from saved files and test on various sentences
"""

import pickle
from pathlib import Path
from sentence_piece import SentencePieceTrainer


class TokenizerTester:
    """Load and test trained tokenizer"""
    
    def __init__(self, model_dir='./tokenizer_models'):
        """
        Args:
            model_dir: Directory containing saved tokenizer models
        """
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        
    def load_tokenizer(self):
        """Load trained SentencePiece tokenizer"""
        print("=" * 80)
        print("LOADING TOKENIZER")
        print("=" * 80)
        
        sp_path = self.model_dir / 'sentencepiece_trainer.pkl'
        
        if not sp_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {sp_path}")
        
        print(f"\n✓ Loading from: {sp_path}")
        
        with open(sp_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"✓ Tokenizer loaded successfully!")
        print(f"   - Vocabulary size: {self.tokenizer.vocab_size:,}")
        print(f"   - Max token length: {self.tokenizer.maxlen}")
    
    def tokenize(self, text, nbest_size=1, show_details=True):
        """
        Tokenize text
        
        Args:
            text: Input text
            nbest_size: Number of best tokenizations to sample from
            show_details: Whether to print details
            
        Returns:
            List of tokens
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        tokens = self.tokenizer.tokenize(text, nbest_size=nbest_size)
        
        if show_details:
            tokens_display = [t.replace('_', '▁') for t in tokens]
            print(f"\n📝 Input: {text}")
            print(f"   → Tokens ({len(tokens)}): {tokens_display}")
        
        return tokens
    
    def test_english_sentences(self):
        """Test on English sentences"""
        print("\n" + "=" * 80)
        print("TESTING ENGLISH SENTENCES")
        print("=" * 80)
        
        sentences = [
            "Hello world!",
            "This is a test sentence.",
            "Machine translation is amazing.",
            "The quick brown fox jumps over the lazy dog.",
            "Natural Language Processing with deep learning.",
            "I love programming in Python.",
            "Artificial intelligence is transforming the world."
        ]
        
        for sentence in sentences:
            self.tokenize(sentence)
    
    def test_vietnamese_sentences(self):
        """Test on Vietnamese sentences"""
        print("\n" + "=" * 80)
        print("TESTING VIETNAMESE SENTENCES")
        print("=" * 80)
        
        sentences = [
            "Xin chào thế giới!",
            "Đây là một câu thử nghiệm.",
            "Dịch máy là một công nghệ tuyệt vời.",
            "Tôi yêu lập trình bằng Python.",
            "Trí tuệ nhân tạo đang thay đổi thế giới.",
            "Học máy và học sâu rất thú vị.",
            "Đại học Bách Khoa Hà Nội là trường đại học hàng đầu Việt Nam."
        ]
        
        for sentence in sentences:
            self.tokenize(sentence)
    
    def test_mixed_sentences(self):
        """Test on mixed language sentences"""
        print("\n" + "=" * 80)
        print("TESTING MIXED LANGUAGE SENTENCES")
        print("=" * 80)
        
        sentences = [
            "I study at Đại học Bách Khoa.",
            "Machine Learning là một lĩnh vực của AI.",
            "Python is my favorite ngôn ngữ lập trình.",
            "NLP (Natural Language Processing) là xử lý ngôn ngữ tự nhiên."
        ]
        
        for sentence in sentences:
            self.tokenize(sentence)
    
    def test_edge_cases(self):
        """Test on edge cases"""
        print("\n" + "=" * 80)
        print("TESTING EDGE CASES")
        print("=" * 80)
        
        cases = [
            "a",
            "ABC",
            "123",
            "!!!",
            "hello@email.com",
            "C++ và Python",
            "GPT-4 và Claude-3",
            "2024-12-01"
        ]
        
        for case in cases:
            self.tokenize(case)
    
    def compare_tokenizations(self, text, nbest_sizes=[1, 3, 5]):
        """
        Compare different n-best tokenizations
        
        Args:
            text: Input text
            nbest_sizes: List of n-best sizes to try
        """
        print("\n" + "=" * 80)
        print("COMPARING N-BEST TOKENIZATIONS")
        print("=" * 80)
        print(f"\n📝 Input: {text}\n")
        
        for nbest in nbest_sizes:
            print(f"N-best = {nbest}:")
            # Sample multiple times to see variation
            for i in range(3):
                tokens = self.tokenizer.tokenize(text, nbest_size=nbest)
                tokens_display = [t.replace('_', '▁') for t in tokens]
                print(f"   Sample {i+1}: {tokens_display}")
            print()
    
    def batch_tokenize(self, texts):
        """
        Tokenize multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of tokenized sequences
        """
        results = []
        for text in texts:
            tokens = self.tokenize(text, show_details=False)
            results.append(tokens)
        return results
    
    def analyze_vocabulary(self):
        """Analyze vocabulary statistics"""
        print("\n" + "=" * 80)
        print("VOCABULARY ANALYSIS")
        print("=" * 80)
        
        # Read vocabulary file if exists
        vocab_path = self.model_dir / 'vocabulary.txt'
        if not vocab_path.exists():
            print("Vocabulary file not found.")
            return
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header lines
        vocab_lines = [l for l in lines if not l.startswith('#') and l.strip()]
        
        print(f"\n📊 Vocabulary Statistics:")
        print(f"   - Total tokens: {len(vocab_lines):,}")
        
        # Analyze token lengths
        token_lengths = []
        for line in vocab_lines:
            token = line.split('\t')[0]
            token_lengths.append(len(token))
        
        print(f"   - Average token length: {sum(token_lengths)/len(token_lengths):.2f}")
        print(f"   - Min token length: {min(token_lengths)}")
        print(f"   - Max token length: {max(token_lengths)}")
        
        # Show top 20 tokens
        print(f"\n🔝 Top 20 Most Frequent Tokens:")
        for i, line in enumerate(vocab_lines[:20], 1):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                token, count = parts
                print(f"   {i:2d}. '{token}' → {count}")
    
    def run_all_tests(self):
        """Run all test suites"""
        self.load_tokenizer()
        self.test_english_sentences()
        self.test_vietnamese_sentences()
        self.test_mixed_sentences()
        self.test_edge_cases()
        
        # Compare n-best tokenizations
        self.compare_tokenizations(
            "Vào chủ nhật ngày 1-9-2019, cơn bão Dorian, một trong những cơn bão mạnh nhất được ghi nhận ở Đại Tây Dương, với sức gió 362 km/h đổ bộ vào đảo Great Abaco, miền bắc Bahamas.",
            nbest_sizes=[1, 3, 5]
        )
        
        self.analyze_vocabulary()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 80)


def main():
    """Main entry point"""
    
    # Configuration
    MODEL_DIR = './tokenizer_models'
    
    # Create tester
    tester = TokenizerTester(model_dir=MODEL_DIR)
    
    # Run all tests
    tester.run_all_tests()
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter text to tokenize (or 'quit' to exit):\n")
    
    while True:
        try:
            text = input(">>> ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text.strip():
                tester.tokenize(text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Goodbye!")


if __name__ == '__main__':
    main()
