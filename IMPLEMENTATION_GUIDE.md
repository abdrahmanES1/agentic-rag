# Moroccan RAG v12: Step-by-Step Implementation
## From Zero to Production in One Weekend

**Your Setup:** RTX 3060 12GB, Qwen2.5-VL quantized  
**Goal:** Upgrade OCR from Tesseract (CER 0.54) → Qwen (CER 0.42)

---

## Saturday Morning (2-3 hours): Installation & Testing

### Step 1: Backup Everything (5 min)

```bash
# Backup current system
cp -r kb_v11 kb_v11_backup_$(date +%Y%m%d)
cp moroccan_rag_v11.py moroccan_rag_v11.py.backup
cp api_v2.py api_v2.py.backup 2>/dev/null || true

# Confirm backups
ls -lh kb_v11_backup_*/
ls -lh *.backup

echo "✓ Backups created"
```

### Step 2: Install Prerequisites (15-20 min)

```bash
# Make installation script executable
chmod +x install_v12.sh

# Run installation
./install_v12.sh

# This will:
# - Check GPU (RTX 3060 12GB)
# - Install Ollama
# - Download Qwen2.5-VL quantized (~8GB, 10-15 min download)
# - Test model
# - Install Python packages
```

**Expected output:**
```
✓ GPU check passed
✓ Disk space check passed
✓ Ollama installed
✓ Qwen quantized model downloaded
✓ Model responds correctly
✓ VRAM usage normal (~7GB)
✓ Python packages installed
✓ Installation Complete!
```

### Step 3: Test GPU Memory (5 min)

```bash
# Test Qwen model VRAM usage
python test_gpu_memory.py
```

**Expected output:**
```
Initial VRAM: 256MB
Loading Qwen...
VRAM after load: 7168MB
VRAM increase: +6912MB

✓ GOOD: Using 5-8GB VRAM
✓ RTX 3060 12GB has enough headroom
```

**If you see >10GB:** You pulled the wrong model!
```bash
# Fix:
ollama rm qwen2.5-vl:7b
ollama pull qwen2.5-vl:7b-q4_0  # Note: q4_0 suffix!
# Retest: python test_gpu_memory.py
```

### Step 4: Test OCR Quality (10 min)

```bash
# Pick a test PDF (preferably scanned or low-quality)
python test_qwen_ocr.py ./pdfs/your_test_file.pdf

# Or test specific page:
python test_qwen_ocr.py ./pdfs/your_test_file.pdf 3
```

**Expected output:**
```
Testing: your_test_file.pdf
Page: 1

1. Digital text extraction:
   ⚠️  Low confidence - OCR recommended

2. Rendering page to image...
   ✓ Image: 1600x2400 pixels

3. Testing Tesseract OCR (v11)...
   Tesseract:
     Characters: 450
     Quality score: 0.42
     Broken words: 25

4. Testing Qwen OCR (v12)...
   ✓ Completed in 4.2 seconds
   Qwen:
     Characters: 520
     Quality score: 0.68
     Broken words: 5

COMPARISON:
✓ Qwen is 61.9% better

Tesseract preview:
الحصول على البطاقة الوطنيه للتعريف ...

Qwen preview:
الحصول على البطاقة الوطنية للتعريف الإلكترونية ...
```

**Decision point:**
- ✓ If Qwen better → Continue to Step 5
- ✗ If Qwen worse → Debug (check model, try different PDF)

---

## Saturday Afternoon (1-2 hours): Code Integration

### Step 5: Add v12 Code (15 min)

```bash
# Open moroccan_rag_v11.py in your editor
nano moroccan_rag_v11.py
# or
code moroccan_rag_v11.py
# or
vim moroccan_rag_v11.py
```

**Changes to make:**

1. **Add imports at top** (if not already there):
   ```python
   # Around line 70-90, add:
   import base64  # For Qwen image encoding
   ```

2. **Add two new functions after `_ocr_from_bytes()`** (line 1514):
   ```python
   # Copy from v12_code_patch.py:
   # - _calculate_ocr_confidence()
   # - _ocr_with_qwen()
   ```

3. **Replace `_load_pdf_fitz()` function** (lines 1470-1493):
   ```python
   # Copy replacement from v12_code_patch.py
   ```

**Quick way (copy-paste):**
```bash
# Open both files side-by-side and copy the functions:
# 1. v12_code_patch.py (source)
# 2. moroccan_rag_v11.py (destination)

# Or use the patch file I created:
cat v12_code_patch.py  # Review the changes
# Then manually add to moroccan_rag_v11.py
```

### Step 6: Verify Code Changes (5 min)

```bash
# Check syntax
python -m py_compile moroccan_rag_v11.py

# If errors:
# - Fix syntax issues
# - Check indentation (Python is strict!)
# - Verify imports

# If OK:
echo "✓ Code syntax valid"
```

### Step 7: Test Modified Code (10 min)

```bash
# Test that pipeline imports correctly
python -c "from moroccan_rag_v11 import MoroccanRAGPipeline; print('✓ Import successful')"

# Test that new functions exist
python -c "from moroccan_rag_v11 import _ocr_with_qwen, _calculate_ocr_confidence; print('✓ New functions found')"

# Quick OCR test
python -c "
from moroccan_rag_v11 import _load_pdf_fitz
from pathlib import Path

pages = _load_pdf_fitz(Path('./pdfs/test.pdf'))
print(f'✓ Loaded {len(pages)} pages')
print(f'First page: {pages[0][\"text\"][:100]}...')
"
```

**Expected:** No errors, pages loaded with Qwen OCR.

---

## Saturday Evening or Sunday (2-4 hours): Build & Test v12 KB

### Step 8: Build v12 KB (1-3 hours depending on corpus size)

```bash
# Build KB in new directory (parallel to v11)
python build_kb_v12.py ./pdfs

# This will:
# 1. Check prerequisites
# 2. Count PDFs
# 3. Build KB using Qwen OCR
# 4. Save to ./kb_v12_test/
```

**What to expect:**
```
Found 47 PDF files in ./pdfs

Continue? [y/N]: y

Building v12 Knowledge Base
Input:  ./pdfs
Output: ./kb_v12_test

Initializing pipeline...
  ✓ Models loaded

Building knowledge base...

Processing: CNIE_dahir_04-20.pdf
  Page 1: PyMuPDF digital text (conf=0.92)
  Page 2: Qwen OCR (conf=0.71)
  Page 3: Qwen OCR (conf=0.68)
  ...

OCR Summary for CNIE_dahir_04-20.pdf:
  Total pages: 15
  PyMuPDF (digital): 8
  Qwen OCR (quantized): 6
  Tesseract (fallback): 1

[... continues for all PDFs ...]

✓ KB built in 45.3 minutes

Build Statistics:
Arabic chunks:  1,234
French chunks:  567
Total chunks:   1,801
Build time:     45.3 minutes

Files created:
  arabic.faiss                  12.45 MB
  arabic_chunks.pkl              3.21 MB
  arabic_bm25.pkl                1.05 MB
  french.faiss                   5.67 MB
  french_chunks.pkl              1.89 MB
  french_bm25.pkl                0.45 MB
  unified.faiss                 18.12 MB
  all_chunks.pkl                 5.10 MB
```

**Monitor GPU during build:**
```bash
# In another terminal:
watch -n 1 nvidia-smi

# Expected:
# - VRAM usage spikes to ~7GB during OCR
# - Drops to ~2GB during embedding
# - Should stay < 12GB always
```

**If CUDA OOM error:**
```bash
# Model is too big. Check:
ollama list

# Should see: qwen2.5-vl:7b-q4_0 (quantized)
# If you see: qwen2.5-vl:7b (full precision)
# → You need to use the quantized version!
```

### Step 9: Quick Quality Check (10 min)

```bash
# Load v12 KB and test a query
python -c "
from moroccan_rag_v11 import MoroccanRAGPipeline

# Initialize with v12 KB
pipeline = MoroccanRAGPipeline()
pipeline.kb.load('./kb_v12_test')

# Test query
result = pipeline.process_question('ما هي الوثائق المطلوبة للبطاقة الوطنية؟')

print('Answer:', result.answer)
print('Sources:', len(result.sources))
print('Grounded:', result.is_grounded)
"
```

**Expected:**
```
Answer: للحصول على البطاقة الوطنية للتعريف الإلكترونية، تحتاج إلى: شهادة الميلاد الكاملة، صورتان فوتوغرافيتان، إثبات الإقامة...

Sources: 3
Grounded: True
```

### Step 10: Visual Comparison (15 min)

```bash
# Compare v11 vs v12 text extraction on same PDF
# (Use the test script from earlier)

# v11 (Tesseract):
git checkout moroccan_rag_v11.py.backup
python test_qwen_ocr.py ./pdfs/problematic.pdf > v11_output.txt

# v12 (Qwen):
git checkout moroccan_rag_v11.py  # Use modified version
python test_qwen_ocr.py ./pdfs/problematic.pdf > v12_output.txt

# Compare:
diff v11_output.txt v12_output.txt
# or
vimdiff v11_output.txt v12_output.txt
```

**Look for:**
- ✓ Fewer broken words in v12
- ✓ Correct diacritics (ة not ه)
- ✓ Better Arabic character recognition
- ⚠️ French might be slightly worse (acceptable, Arabic is priority)

---

## Sunday Evening: Production Deployment

### Step 11: Create Test Queries (30 min)

Create a file `test_queries.txt`:
```
ما هي الوثائق المطلوبة للبطاقة الوطنية؟
كم مدة صلاحية جواز السفر؟
ما هي رسوم البطاقة الوطنية؟
Comment obtenir un passeport biométrique?
Quelle est la durée de validité de la carte d'identité?
```

Run evaluation:
```bash
# Test v11 KB
python -c "
from moroccan_rag_v11 import MoroccanRAGPipeline

pipeline = MoroccanRAGPipeline()
pipeline.kb.load('./kb_v11')

with open('test_queries.txt') as f:
    for i, query in enumerate(f, 1):
        result = pipeline.process_question(query.strip())
        print(f'{i}. {result.is_grounded}')
" > v11_results.txt

# Test v12 KB
python -c "
from moroccan_rag_v11 import MoroccanRAGPipeline

pipeline = MoroccanRAGPipeline()
pipeline.kb.load('./kb_v12_test')

with open('test_queries.txt') as f:
    for i, query in enumerate(f, 1):
        result = pipeline.process_question(query.strip())
        print(f'{i}. {result.is_grounded}')
" > v12_results.txt

# Compare
paste v11_results.txt v12_results.txt
```

**Expected:**
```
v11    v12
True   True
True   True
False  True  ← v12 improvement
True   True
True   True
```

### Step 12: Deploy to Production (10 min)

**If v12 ≥ v11 quality:**

```bash
# Stop API
sudo systemctl stop moroccan-rag-api

# Blue-green deployment (zero-downtime alternative)
# Option A: Symlink switch (instant rollback)
mv kb_current kb_v11_backup  # Keep old KB
ln -s kb_v12_test kb_current  # Point to new KB

# Option B: Direct replacement (slower rollback)
# mv kb_v11 kb_v11_backup
# mv kb_v12_test kb_v11

# Start API
sudo systemctl start moroccan-rag-api

# Test
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي الوثائق المطلوبة للبطاقة الوطنية؟"}'

# Should return answer
```

### Step 13: Monitor Production (24 hours)

```bash
# Watch logs
tail -f /var/log/moroccan-rag.log

# Monitor GPU
watch -n 5 nvidia-smi

# Check query success rate
grep "is_grounded.*True" /var/log/moroccan-rag.log | wc -l
grep "is_grounded.*False" /var/log/moroccan-rag.log | wc -l
```

**If issues occur:**

```bash
# INSTANT ROLLBACK (< 30 seconds):
sudo systemctl stop moroccan-rag-api
rm kb_current
ln -s kb_v11_backup kb_current
sudo systemctl start moroccan-rag-api

# Investigate issues
# Fix and retry
```

---

## Week 2: Cleanup & Optimization

### Step 14: Finalize (If v12 Stable)

After 1 week of stable v12 operation:

```bash
# Remove v11 backup
rm -rf kb_v11_backup_*

# Document changes
cat > CHANGELOG.md << 'EOF'
# v12 Upgrade - $(date +%Y-%m-%d)

## Changes:
- Upgraded OCR: Tesseract → Qwen2.5-VL (quantized)
- Arabic CER: 0.54 → 0.42 (22% improvement)
- VRAM usage: ~7GB (fits RTX 3060 12GB)

## Results:
- Query success rate: +5-10%
- Visual quality: Significantly better Arabic text
- Cost: $0 API fees (self-hosted)

## Rollback:
Kept in: moroccan_rag_v11.py.backup
EOF

echo "✓ v12 deployment successful!"
```

---

## Troubleshooting Guide

### Issue: "CUDA out of memory"

**Diagnosis:**
```bash
nvidia-smi  # Check VRAM usage
ollama list | grep qwen  # Check model name
```

**Solutions:**
1. Wrong model (full precision instead of quantized):
   ```bash
   ollama rm qwen2.5-vl:7b
   ollama pull qwen2.5-vl:7b-q4_0
   ```

2. Too many models loaded:
   ```bash
   # Unload all Ollama models
   pkill ollama
   systemctl restart ollama
   ```

3. Other GPU processes:
   ```bash
   # Check what's using GPU
   nvidia-smi
   # Kill if necessary
   ```

### Issue: "Qwen returns empty text"

**Diagnosis:**
```bash
# Test Qwen directly
ollama run qwen2.5-vl:7b-q4_0
```

**Solutions:**
1. Model not loaded:
   ```bash
   ollama pull qwen2.5-vl:7b-q4_0
   ```

2. Ollama service not running:
   ```bash
   systemctl start ollama
   ```

3. Image too large:
   ```python
   # In _ocr_with_qwen(), reduce resolution:
   mat = pymupdf.Matrix(1.5, 1.5)  # Instead of 2.0
   ```

### Issue: "v12 quality worse than v11"

**Investigation:**
```bash
# Compare specific pages
python test_qwen_ocr.py ./pdfs/problem_file.pdf
```

**Possible causes:**
1. French-heavy document (Tesseract better for French)
2. Very clean scan (PyMuPDF digital extraction better)
3. Unusual layout (tables, diagrams)

**Solutions:**
- Adjust confidence threshold in `_load_pdf_fitz()`
- For French docs: Keep using Tesseract (modify cascade)
- For tables: Add table extraction (Phase 2)

---

## Success Criteria

✓ **Installation:** All prerequisites installed, models loaded  
✓ **GPU:** VRAM usage ~7GB (< 12GB limit)  
✓ **OCR Quality:** Visual inspection shows cleaner Arabic  
✓ **KB Build:** Completes without errors  
✓ **Retrieval:** Test queries return grounded answers  
✓ **Production:** Stable for 1 week, no regressions  

**If all criteria met → v12 deployment successful! 🎉**

---

## Summary

**Time investment:**
- Saturday: 3-5 hours (installation + code changes)
- Sunday: 2-4 hours (KB build + testing)
- Total: 1 weekend

**Results:**
- 22% better Arabic OCR (CER 0.54 → 0.42)
- Cleaner text extraction
- Zero API costs
- Fits RTX 3060 12GB

**Next steps:**
- Phase 2: Add table extraction (optional)
- Phase 3: Add visual retrieval (optional)
- Optimization: Context caching, monitoring

**You're done with the core v12 upgrade!**
