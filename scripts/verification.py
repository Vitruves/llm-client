#!/usr/bin/env python3

import argparse
import pandas as pd
import sys
import signal
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
	logger = logging.getLogger(__name__)
	handler = logging.StreamHandler()
	
	if debug:
		level = logging.DEBUG
		fmt = '%(asctime)s - DEBUG : %(message)s'
	elif verbose:
		level = logging.INFO
		fmt = '%(asctime)s - %(levelname)s : %(message)s'
	else:
		level = logging.WARNING
		fmt = '%(asctime)s - %(levelname)s : %(message)s'
	
	logging.basicConfig(level=level, format=fmt, datefmt='%H:%M')
	return logger

def clear_console():
	os.system('clear' if os.name == 'posix' else 'cls')

class ManualVerifier:
	def __init__(self, input_file: Path, comment_col: str, prediction_col: str, 
				 output_file: Path, resume_file: Optional[Path] = None, logger: logging.Logger = None):
		self.input_file = input_file
		self.comment_col = comment_col
		self.prediction_col = prediction_col
		self.output_file = output_file
		self.resume_file = resume_file
		self.logger = logger or logging.getLogger(__name__)
		self.df = None
		self.verified_df = None
		self.current_index = 0
		self.verification_col = 'manual_verification'
		
	def load_data(self) -> None:
		self.logger.info(f"Loading data from {self.input_file}")
		try:
			self.df = pd.read_parquet(self.input_file)
			self.logger.info(f"Loaded {len(self.df)} records")
		except Exception as e:
			self.logger.error(f"Failed to load input file: {e}")
			sys.exit(1)
			
		if self.comment_col not in self.df.columns:
			self.logger.error(f"Comment column '{self.comment_col}' not found in data")
			sys.exit(1)
			
		if self.prediction_col not in self.df.columns:
			self.logger.error(f"Prediction column '{self.prediction_col}' not found in data")
			sys.exit(1)
	
	def load_resume_data(self) -> None:
		if not self.resume_file or not self.resume_file.exists():
			self.verified_df = self.df.copy()
			self.verified_df[self.verification_col] = None
			return
			
		self.logger.info(f"Resuming from {self.resume_file}")
		try:
			self.verified_df = pd.read_parquet(self.resume_file)
			verified_count = self.verified_df[self.verification_col].notna().sum()
			self.current_index = verified_count
			self.logger.info(f"Resuming from record {self.current_index + 1} ({verified_count} already verified)")
		except Exception as e:
			self.logger.error(f"Failed to load resume file: {e}")
			sys.exit(1)
	
	def display_record(self, index: int) -> None:
		clear_console()
		
		record = self.verified_df.iloc[index]
		comment = str(record[self.comment_col])
		prediction = record[self.prediction_col]
		
		verified_count = self.verified_df[self.verification_col].notna().sum()
		total_records = len(self.verified_df)
		progress = f"({verified_count}/{total_records} verified)"
		
		print(f"Record {index + 1} of {total_records} {progress}")
		print("=" * 80)
		print(f"{comment}")
		print(f"\nPrediction: {prediction}")
		print("=" * 80)
	
	def get_user_input(self) -> str:
		while True:
			response = input("Is the prediction correct? [y/N/q]: ").strip().lower()
			if response in ['y', 'n', '', 'q']:
				return response
			print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")
	
	def save_progress(self) -> None:
		try:
			self.verified_df.to_parquet(self.output_file)
			self.logger.info(f"Progress saved to {self.output_file}")
		except Exception as e:
			self.logger.error(f"Failed to save progress: {e}")
	
	def run_verification(self) -> None:
		self.load_data()
		self.load_resume_data()
		
		total_records = len(self.verified_df)
		
		def signal_handler(signum, frame):
			clear_console()
			self.logger.info("Interrupted by user. Saving progress...")
			self.save_progress()
			sys.exit(0)
		
		signal.signal(signal.SIGINT, signal_handler)
		
		try:
			for i in range(self.current_index, total_records):
				if self.verified_df.iloc[i][self.verification_col] is not None:
					continue
					
				self.display_record(i)
				response = self.get_user_input()
				
				if response == 'q':
					clear_console()
					self.logger.info("Verification stopped by user")
					break
				elif response == 'y':
					self.verified_df.iloc[i, self.verified_df.columns.get_loc(self.verification_col)] = True
				elif response in ['n', '']:
					self.verified_df.iloc[i, self.verified_df.columns.get_loc(self.verification_col)] = False
				
				if (i + 1) % 10 == 0:
					self.save_progress()
					self.logger.debug(f"Auto-saved progress at record {i + 1}")
			
			clear_console()
			self.save_progress()
			
			verified_count = self.verified_df[self.verification_col].notna().sum()
			correct_count = self.verified_df[self.verification_col].sum() if verified_count > 0 else 0
			
			self.logger.info(f"Verification complete: {verified_count}/{total_records} records verified")
			if verified_count > 0:
				accuracy = (correct_count / verified_count) * 100
				self.logger.info(f"Accuracy: {correct_count}/{verified_count} ({accuracy:.1f}%)")
				
		except Exception as e:
			clear_console()
			self.logger.error(f"Error during verification: {e}")
			self.save_progress()
			sys.exit(1)

def main():
	parser = argparse.ArgumentParser(
		description="Manually verify prediction accuracy on pharmaceutical review data",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  %(prog)s -i reviews.parquet --comment-col comment --prediction-col predicted_label -o verified.parquet
  %(prog)s -i data.parquet --comment-col comment --prediction-col prediction -o output.parquet --resume partial.parquet
		"""
	)
	
	parser.add_argument('-i', '--input', type=Path, required=True,
						help='Input parquet file containing reviews and predictions')
	parser.add_argument('--comment-col', type=str, required=True,
						help='Name of the column containing review comments')
	parser.add_argument('--prediction-col', type=str, required=True,
						help='Name of the column containing predictions to verify')
	parser.add_argument('-o', '--output', type=Path, required=True,
						help='Output parquet file to save verification results')
	parser.add_argument('--resume', type=Path,
						help='Resume verification from partially completed file')
	parser.add_argument('-v', '--verbose', action='store_true',
						help='Enable verbose output (INFO level)')
	parser.add_argument('--debug', action='store_true',
						help='Enable debug output (DEBUG level)')
	
	args = parser.parse_args()
	
	logger = setup_logging(args.verbose, args.debug)
	
	if not args.input.exists():
		logger.error(f"Input file not found: {args.input}")
		sys.exit(1)
	
	if args.resume and not args.resume.exists():
		logger.warning(f"Resume file not found: {args.resume}")
		args.resume = None
	
	args.output.parent.mkdir(parents=True, exist_ok=True)
	
	verifier = ManualVerifier(
		input_file=args.input,
		comment_col=args.comment_col,
		prediction_col=args.prediction_col,
		output_file=args.output,
		resume_file=args.resume,
		logger=logger
	)
	
	verifier.run_verification()

if __name__ == '__main__':
	main()