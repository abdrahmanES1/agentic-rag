#!/usr/bin/env python3
"""
Build v12 Knowledge Base with Qwen OCR.
Runs in parallel to v11 (blue-green deployment).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import shutil
from pathlib import Path
from datetime import datetime
import time

def check_prerequisites():
    """Verify system is ready for v12 build"""
    print("Checking prerequisites...")
    issues = []
    
    # Check if v12 code modifications are in place
    try:
        # Try to import the modified code
        sys.path.insert(0, str(Path.cwd()))
        import moroccan_rag_v12 as moroccan_rag_v12
        
        # Check if new functions exist
        if not hasattr(moroccan_rag_v12, '_ocr_with_qwen'):
            issues.append("_ocr_with_qwen() function not found")
        if not hasattr(moroccan_rag_v12, '_calculate_ocr_confidence'):
            issues.append("_calculate_ocr_confidence() function not found")
        
        print("  ✓ v12 code modifications detected")
    except Exception as e:
        issues.append(f"Cannot import moroccan_rag_v11.py: {e}")
    
    # Check if Ollama is running
    import subprocess
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("  ✓ Ollama service running")
        else:
            issues.append("Ollama service not responding")
    except Exception as e:
        issues.append(f"Cannot reach Ollama: {e}")
    
    # Check if Qwen model exists
    # try:
    #     result = subprocess.run(
    #         ["ollama", "list"],
    #         capture_output=True,
    #         text=True
    #     )
    #     if "qwen/qwen3-vl-8b" in result.stdout or "qwen2.5-vl" in result.stdout:
    #         print("  ✓ Qwen model installed")
    #     else:
    #         issues.append("Qwen model not found. Run: ollama pull qwen2.5-vl:7b-q4_0")
    # except Exception as e:
    #     issues.append(f"Cannot check Ollama models: {e}")
    
    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"  ✓ GPU detected: {gpu_info}")
        else:
            print("  ⚠️  Cannot detect GPU (will be slow on CPU)")
    except:
        print("  ⚠️  nvidia-smi not available")
    
    # Check disk space
    stat = shutil.disk_usage('.')
    free_gb = stat.free / (1024**3)
    print(f"  Disk space: {free_gb:.1f}GB free")
    if free_gb < 10:
        issues.append(f"Low disk space: {free_gb:.1f}GB (need at least 10GB)")
    
    return issues

def build_kb(pdf_dir: str, output_dir: str):
    """Build v12 KB using modified pipeline"""
    print(f"\n{'='*70}")
    print("Building v12 Knowledge Base")
    print(f"{'='*70}\n")
    
    print(f"Input:  {pdf_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Import pipeline
    from moroccan_rag_v12 import MoroccanRAGPipeline
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = MoroccanRAGPipeline()
    print("  ✓ Models loaded")
    print()
    
    # Build KB
    print("Building knowledge base...")
    print("This will use:")
    print("  - PyMuPDF for digital text")
    print("  - Qwen OCR for scanned pages")
    print("  - Tesseract as fallback")
    print()
    
    start_time = time.time()
    
    try:
        pipeline.kb = pipeline.setup()
        pipeline.kb = pipeline.build_knowledge_base(pdf_dir)
        
        build_time = time.time() - start_time
        
        print()
        print(f"✓ KB built in {build_time/60:.1f} minutes")
        print()
        
        # Save KB
        print(f"Saving to {output_dir}...")
        pipeline.build_knowledge_base(output_dir, force_rebuild=False)
        print("  ✓ KB saved")
        
        # Print stats
        print()
        print("=" * 70)
        print("Build Statistics")
        print("=" * 70)
        print(f"Arabic chunks:  {len(pipeline.kb.arabic_chunks)}")
        print(f"French chunks:  {len(pipeline.kb.french_chunks)}")
        print(f"Total chunks:   {len(pipeline.kb.all_chunks)}")
        print(f"Build time:     {build_time/60:.1f} minutes")
        print()
        
        # Check files
        files_created = list(Path(output_dir).glob("*"))
        print("Files created:")
        for f in sorted(files_created):
            size_mb = f.stat().st_size / (1024**2)
            print(f"  {f.name:<30} {size_mb:>8.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_v11(v11_dir: str, v12_dir: str):
    """Compare v11 and v12 KB stats"""
    print(f"\n{'='*70}")
    print("Comparing v11 vs v12")
    print(f"{'='*70}\n")
    
    try:
        import pickle
        
        # Load v11 chunks
        v11_ar = pickle.load(open(f"{v11_dir}/arabic_chunks.pkl", "rb"))
        v11_fr = pickle.load(open(f"{v11_dir}/french_chunks.pkl", "rb"))
        
        # Load v12 chunks
        v12_ar = pickle.load(open(f"{v12_dir}/arabic_chunks.pkl", "rb"))
        v12_fr = pickle.load(open(f"{v12_dir}/french_chunks.pkl", "rb"))
        
        print(f"                      v11        v12        Change")
        print(f"-" * 55)
        print(f"Arabic chunks:    {len(v11_ar):>6}     {len(v12_ar):>6}     {len(v12_ar)-len(v11_ar):>+6}")
        print(f"French chunks:    {len(v11_fr):>6}     {len(v12_fr):>6}     {len(v12_fr)-len(v11_fr):>+6}")
        print(f"Total:            {len(v11_ar)+len(v11_fr):>6}     {len(v12_ar)+len(v12_fr):>6}     {(len(v12_ar)+len(v12_fr))-(len(v11_ar)+len(v11_fr)):>+6}")
        print()
        
        # Sample comparison
        print("Sample Arabic chunk comparison:")
        print()
        print("v11 (first chunk):")
        print(v11_ar[0].text[:200])
        print()
        print("v12 (first chunk):")
        print(v12_ar[0].text[:200])
        
    except Exception as e:
        print(f"⚠️  Cannot compare: {e}")

def main():
    print(f"\n🔧 Moroccan RAG v12 KB Builder")
    print(f"{'='*70}\n")
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("\n❌ Prerequisites check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix these issues before building KB.")
        sys.exit(1)
    
    print("\n✓ All prerequisites met\n")
    
    # Get PDF directory
    if len(sys.argv) > 1:
        pdf_dir = sys.argv[1]
    else:
        pdf_dir = "./pdfs"
    
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"❌ PDF directory not found: {pdf_dir}")
        print("\nUsage: python build_kb_v12.py [pdf_directory]")
        sys.exit(1)
    
    # Count PDFs
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    print()
    
    # Confirm
    print("This will build a NEW KB using v12 (Qwen OCR)")
    print("Your existing v11 KB will NOT be modified")
    print()
    response = input("Continue? [y/N]: ")
    if not response.lower().startswith('y'):
        print("Cancelled.")
        return
    
    # Build
    output_dir = "./kb_v12_test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success = build_kb(str(pdf_path), output_dir)
    
    if success:
        # Save build log
        log_file = f"{output_dir}/build_log_{timestamp}.txt"
        with open(log_file, 'w') as f:
            f.write(f"v12 KB Build Log\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"PDF directory: {pdf_path}\n")
            f.write(f"Number of PDFs: {len(pdf_files)}\n")
            f.write(f"Output directory: {output_dir}\n")
        
        print(f"\n✓ Build log saved: {log_file}")
        
        # Compare with v11 if exists
        if Path("./kb_v11").exists():
            compare_with_v11("./kb_v11", output_dir)
        
        print(f"\n{'='*70}")
        print("Next Steps:")
        print(f"{'='*70}")
        print()
        print("1. Test v12 KB:")
        print(f"   python test_queries.py --kb {output_dir}")
        print()
        print("2. Compare quality:")
        print("   python evaluate_kb.py --v11 ./kb_v11 --v12 ./kb_v12_test")
        print()
        print("3. If v12 is better, deploy:")
        print("   mv kb_current kb_v11_backup")
        print(f"   ln -s {output_dir} kb_current")
        print("   systemctl restart moroccan-rag-api")
        print()
        print("4. Rollback if needed:")
        print("   rm kb_current")
        print("   ln -s kb_v11_backup kb_current")
        print("   systemctl restart moroccan-rag-api")
        print()
    else:
        print("\n❌ Build failed. Check logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
