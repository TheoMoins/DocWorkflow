from src.tasks.base_tasks import BaseTask
from abc import abstractmethod
import os
import glob
from tqdm import tqdm
from pathlib import Path
from lxml import etree as ET
from src.tasks.htr.postprocessing import clean_alto_file

from jiwer import cer, wer

from src.utils.metrics import calculate_htr_metrics
from src.alto.alto_text import read_document_text, read_lines_text, deduplicate_alto_consecutive_lines

class BaseHTR(BaseTask):
    """
    Base class for HTR (Handwritten Text Recognition) tasks.
    Provides common functionality for all HTR implementations.
    """
    
    def __init__(self, config):
        """
        Initialize the HTR base class.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
    
    @abstractmethod
    def load(self):
        """Load the model. Must be implemented by subclasses."""
        pass
    
    def train(self, data_path=None, **kwargs):
        """
        Train the HTR model. Default implementation prints a message.
        Can be overridden by subclasses that support training.
        """
        print(f"Training for {self.name} is not yet implemented.")
    


    def predict(self, data_path, output_dir, save_image=True, **kwargs):
        """
        Override predict to apply post-processing deduplication on all output ALTO files.
        """
        results = super().predict(data_path=data_path, output_dir=output_dir, save_image=save_image, **kwargs)

        # Deduplicate consecutive lines in all produced ALTO files
        for alto_path in glob.glob(str(Path(output_dir) / '**' / '*.xml'), recursive=True):
            try:
                deduplicate_alto_consecutive_lines(alto_path)
            except Exception as e:
                print(f"  Warning: deduplication failed on {alto_path}: {e}")

            try:
                clean_alto_file(alto_path)
            except Exception as e:
                print(f"  Warning: CATMuS cleaning failed on {alto_path}: {e}")

        return results
    
    def _score_batch(self, pred_files, gt_files, pred_dir, gt_dir):
        """
        Score a batch of HTR predictions.
        """
        all_gt_texts = []
        all_pred_texts = []
        page_scores = []
        
        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="  Scoring", unit="page"):
            try:
                gt_lines = read_lines_text(gt_file)
                pred_lines = read_lines_text(pred_file)
                
                page_gt_texts = []
                page_pred_texts = []
                competition_preds = {}
                competition_gt = {}   
                
                if len(pred_lines) == 1 and len(gt_lines) > 1:
                    # Fallback: split prediction by line breaks
                    single_text = pred_lines[0]['text']
                    pred_texts_split = [t.strip() for t in single_text.split('\n') if t.strip()]
                    
                    # If split matches ground truth line count, use it
                    if len(pred_texts_split) == len(gt_lines):
                        for gt_line, pred_text in zip(gt_lines, pred_texts_split):
                            gt_text = gt_line['text']
                            if gt_text.strip():
                                page_gt_texts.append(gt_text)
                                page_pred_texts.append(pred_text)

                    else:
                        full_gt = ' '.join(line['text'] for line in gt_lines if line['text'].strip())
                        full_pred = ' '.join(pred_texts_split)
                        if full_gt.strip():
                            page_gt_texts.append(full_gt)
                            page_pred_texts.append(full_pred)
                else:
                    gt_text = read_document_text(gt_file)
                    pred_text = read_document_text(pred_file)
                    page_stem = Path(gt_file).stem
                    for i, l in enumerate(gt_lines):
                        if l['text'].strip():
                            competition_gt[f"{page_stem}/{i}"] = l['text']
                    for i, l in enumerate(pred_lines):
                        if i < len(gt_lines) and gt_lines[i]['text'].strip():
                            competition_preds[f"{page_stem}/{i}"] = l['text']

                    if gt_text.strip():
                        page_gt_texts.append(gt_text)
                        page_pred_texts.append(pred_text)
                
                if page_gt_texts and page_pred_texts:
                    all_gt_texts.extend(page_gt_texts)
                    all_pred_texts.extend(page_pred_texts)
                    
                    page_scores.append({
                        'page': Path(gt_file).stem,
                        'cer': cer(page_gt_texts, page_pred_texts),
                        'wer': wer(page_gt_texts, page_pred_texts),
                        'char_count': sum(len(t) for t in page_gt_texts),
                        'word_count': sum(len(t.split()) for t in page_gt_texts)
                    })
            except Exception as e:
                print(f"  Error on {Path(gt_file).name}: {e}")
        
        # Calculate global metrics
        metrics_dict = calculate_htr_metrics(
            all_gt_texts, all_pred_texts, page_scores,
            competition_preds=competition_preds,
            competition_gt=competition_gt
        )
        
        return metrics_dict, page_scores
    
    def _get_score_file_extensions(self):
        """HTR scoring works with XML files, not images."""
        return ['*.xml']