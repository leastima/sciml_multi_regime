#!/usr/bin/env python3
"""
Script to flatten the nested model directory structure for CKA analysis.

Current structure: /scratch/kenzhong/pinnacle/{PDE}/{optimizer}/{exp_name}/model_epoch_XXX.pt
Target structure: /scratch/kenzhong/pinnacle/{PDE}/{optimizer}/{exp_name}_epoch_XXX.pt
"""

import os
import shutil
import glob
import argparse

def flatten_model_directory(source_dir, target_dir, dry_run=False):
    """
    Flatten nested model directories into a flat structure.
    
    Args:
        source_dir: Directory with nested structure
        target_dir: Target flat directory
        dry_run: If True, only print what would be done
    """
    
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)
    
    model_count = 0
    
    # Find all experiment directories
    for exp_dir in glob.glob(os.path.join(source_dir, "*")):
        if not os.path.isdir(exp_dir):
            continue
            
        exp_name = os.path.basename(exp_dir)
        print(f"Processing experiment: {exp_name}")
        
        # Find all .pt files in this experiment directory
        pt_files = glob.glob(os.path.join(exp_dir, "*.pt"))
        
        if not pt_files:
            print(f"  No .pt files found in {exp_dir}")
            continue
            
        for pt_file in pt_files:
            # Extract epoch number from filename if possible
            filename = os.path.basename(pt_file)
            
            # Create new flat filename
            if "epoch_" in filename:
                # File already has epoch info: model_epoch_XXX.pt
                epoch_part = filename.split("epoch_")[1]  # XXX.pt
                new_filename = f"{exp_name}_epoch_{epoch_part}"
            else:
                # File doesn't have epoch info, just add it
                name_part = filename.replace(".pt", "")
                new_filename = f"{exp_name}_{name_part}.pt"
            
            source_path = pt_file
            target_path = os.path.join(target_dir, new_filename)
            
            if dry_run:
                print(f"  Would copy: {source_path} -> {target_path}")
            else:
                print(f"  Copying: {source_path} -> {target_path}")
                shutil.copy2(source_path, target_path)
                
            model_count += 1
    
    print(f"\nTotal models {'would be' if dry_run else ''} processed: {model_count}")
    return model_count

def main():
    parser = argparse.ArgumentParser(description="Flatten nested model directory structure")
    parser.add_argument('--source', 
                       default='/scratch/kenzhong/pinnacle/convection/lbfgs/models',
                       help='Source directory with nested structure')
    parser.add_argument('--target', 
                       default='/scratch/kenzhong/pinnacle/convection/lbfgs/models_flat',
                       help='Target flat directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show what would be done, don\'t actually copy files')
    
    args = parser.parse_args()
    
    print(f"Source directory: {args.source}")
    print(f"Target directory: {args.target}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory {args.source} does not exist!")
        return
    
    # List what's in the source directory
    print("Found experiment directories:")
    exp_dirs = glob.glob(os.path.join(args.source, "*"))
    for exp_dir in exp_dirs:
        if os.path.isdir(exp_dir):
            exp_name = os.path.basename(exp_dir)
            pt_files = glob.glob(os.path.join(exp_dir, "*.pt"))
            print(f"  {exp_name}: {len(pt_files)} .pt files")
    print("-" * 50)
    
    model_count = flatten_model_directory(args.source, args.target, args.dry_run)
    
    if not args.dry_run and model_count > 0:
        print(f"\nFlattening complete! Check {args.target}")
        print("You can now run CKA analysis on the flattened directory.")

if __name__ == "__main__":
    main()
