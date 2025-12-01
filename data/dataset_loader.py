"""Dataset Loading and Preprocessing"""

from datasets import load_dataset
from config.training_config import TrainingConfig

class DatasetLoader:
    """Dataset yükleme ve preprocessing"""
    
    DEEP_DATASET = "Naholav/CodeGen-Deep-5K"
    DIVERSE_DATASET = "Naholav/CodeGen-Diverse-5K"
    
    def __init__(self, dataset_name, tokenizer, use_reasoning=False):
        """
        Args:
            dataset_name: "deep" veya "diverse"
            tokenizer: Model tokenizer
            use_reasoning: output field kullan (reasoning ile) veya solution field (sadece kod)
        """
        self.tokenizer = tokenizer
        self.use_reasoning = use_reasoning
        
        # Dataset seç
        if dataset_name.lower() == "deep":
            self.dataset_path = self.DEEP_DATASET
        elif dataset_name.lower() == "diverse":
            self.dataset_path = self.DIVERSE_DATASET
        else:
            raise ValueError(f"Geçersiz dataset: {dataset_name}")
        
        # System prompt seç
        if use_reasoning:
            self.system_prompt = TrainingConfig.SYSTEM_PROMPT_REASONING
            self.max_length = TrainingConfig.max_length_reasoning
        else:
            self.system_prompt = TrainingConfig.SYSTEM_PROMPT_SOLUTION
            self.max_length = TrainingConfig.max_length_solution
    
    def load_and_prepare(self):
        """Dataset'i yükle ve preprocessing yap"""
        # Dataset yükle
        dataset = load_dataset(self.dataset_path)
        
        # Train/test split
        if "train" not in dataset:
            # Eğer split yoksa, manuel split yap
            dataset = dataset["train"].train_test_split(test_size=0.1, seed=TrainingConfig.seed)
        
        # Preprocessing
        train_dataset = dataset["train"].map(
            self._preprocess_function,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing train data"
        )
        
        test_dataset = dataset["test"].map(
            self._preprocess_function,
            remove_columns=dataset["test"].column_names,
            desc="Preprocessing test data"
        )
        
        return train_dataset, test_dataset
    
    def _preprocess_function(self, examples):
        """
        Her örneği model formatına çevir
        
        Dataset fields:
        - input: Problem açıklaması
        - output: Reasoning trace + kod (<think> tag'leri ile)
        - solution: Sadece temiz kod
        - difficulty: Zorluk seviyesi
        """
        # Field seç
        if self.use_reasoning:
            code_field = examples["output"]
        else:
            code_field = examples["solution"]
        
        # Prompt oluştur
        prompt = f"{self.system_prompt}\n\nProblem:\n{examples['input']}\n\nSolution:\n{code_field}"
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Labels = input_ids (causal LM için)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
