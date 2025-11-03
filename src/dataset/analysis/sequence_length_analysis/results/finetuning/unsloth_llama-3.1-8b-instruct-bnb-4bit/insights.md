✅ GOOD SIGNS:
1. Natural scaling: Longer code → generally more reasoning needed
2. Not too strong (not 0.9+): Shows variation in reasoning depth
3. Wide spread: Different functions need different reasoning for same length
   - 500-token function might need 300-1000 reasoning tokens
   - Depends on complexity, not just length!

💡 INSIGHTS:
- Most functions are short (< 600 tokens)
- Reasoning is typically proportional but with high variance
- Some short functions need extensive reasoning (complex vulnerabilities)
- Some long functions need minimal reasoning (obviously safe/vulnerable)

---
### 2. Reasoning Length vs Answer Length (r = 0.210)

**Pattern Observed:**
- ✅ **Very weak correlation** (0.210) - **this is EXCELLENT!**
- 📍 **Horizontal bands** at specific answer lengths
- 📍 **Four distinct levels**: ~6, 12, 16, 20, 23 tokens
- 📍 **Answer length independent of reasoning length**

**Interpretation:**

✅ EXCELLENT SIGNS:
1. Answer format is CONSISTENT regardless of reasoning complexity
2. Multiple horizontal bands = different answer types with fixed formats
3. Low correlation (0.210) = format rules are followed

💡 BAND ANALYSIS:
Looking at the 4 distinct horizontal bands:

Band 1 (NO):              50% of samples  ← Balanced!
Bands 2-5 (YES):          50% of samples
    Band 2 (1 CWE):       ~25-30%
    Band 3 (2 CWEs):      ~12-15%
    Band 4 (3 CWEs):      \~5-7%
    Band 5 (4+ CWEs):     \~1-3%

Band 1 (~6 tokens):  "Final Answer: NO"
Band 2 (~12 tokens): "Final Answer: YES (CWE-119)"
Band 3 (\~16 tokens): "Final Answer: YES (CWE-119, CWE-787)"
Band 4 (\~20 tokens): "Final Answer: YES (CWE-119, CWE-787, CWE-476)"
Band 5 (\~23 tokens): 3+ CWEs or longer CWE IDs


# The distinct bands in Reasoning vs Answer show:
# - Most samples: 1-2 CWEs (bands 2-3)
# - Few samples: 3+ CWEs (bands 4-5)

# This is REALISTIC for cybersecurity:
# - Buffer overflow (CWE-119) often occurs alone
# - Sometimes coupled with write issues (CWE-787)
# - Rarely 4+ distinct vulnerabilities in single function

# No action needed - distribution looks natural

---

## 🔬 Deep Dive: What the Shapes Tell Us

### Triangular/Cone Shape (Code vs Reasoning)

Why this shape?

    2000 |                    ●
         |                 ●  ●
         |              ●  ●  ●
    1500 |           ●  ●  ●  ●
         |        ●  ●  ●  ●  ●
         |     ●  ●  ●  ●  ●  ●
    1000 |  ●  ●  ●  ●  ●  ●  ●
         |●●●●●●●●●●●●●●●●●●●●●●
     500 |●●●●●●●●●●●●●●●●●●●●●●
         |●●●●●●●●●●●●●●●●●●●●●●
       0 |___________________________
         0    500   1000  1500  2000
                Code Length

Interpretation:
- Base (left): Many short functions, wide reasoning variance
  → Some simple, some hiding complex vulnerabilities
- Middle: Moderate-length code, reasoning somewhat predictable
- Top (right): Few long functions, inherently need more reasoning
- Cone shape: Variance increases with length (natural!)

### Horizontal Bands (Reasoning vs Answer)

Why these bands?

   Answer Tokens
      |
   23 |  ●●              ← ~1.5% of all samples (4+ CWEs)
      |
   20 |  ●●●●●●          ← ~6% of all samples (3 CWEs)
      |
   16 |  ●●●●●●●●●●●●    ← \~15% of all samples (2 CWEs)
      |
   12 |  ●●●●●●●●●●●●●●●●●●●●●●  ← \~27.5% of all samples (1 CWE)
      |
    6 |  ●●●●●●●●●●●●●●●●●●●●●●  ← 50% of all samples (NO)
      |___________________________
          Reasoning Tokens

Band 1 looks denser because:
- It's 50% of samples (largest single group)
- All "NO" answers plot at exactly the same y-value
- Perfect horizontal line = no variation

Band 2 is also dense because:
- It's 27.5% of samples (second largest group)
- Most vulnerable functions have single CWE

Bands 3-5 are sparser because:
- Progressively fewer samples
- Multiple CWEs are less common

Interpretation:
- Perfect horizontal bands = format specification FOLLOWED
- Band width varies = different CWE counts
- No diagonal trend = answer doesn't depend on reasoning length ✓
- This is EXACTLY what you want!

---

## 📝 Summary Report

DATASET CORRELATION ANALYSIS
=============================

✅ Code vs Reasoning (r = 0.513):
   - GOOD: Moderate positive correlation
   - PATTERN: Reasoning scales with code complexity
   - VARIANCE: High (good!) - shows nuanced reasoning
   - ACTION: None needed

✅ Reasoning vs Answer (r = 0.210):
   - EXCELLENT: Very weak correlation
   - PATTERN: Answer format is consistent (horizontal bands)
   - BANDS: 5 distinct levels = NO, 1-CWE, 2-CWE, 3-CWE, 4+-CWE
   - ACTION: None needed

💡 KEY FINDINGS:
   1. Answer tokens: ~2% of total (12.5 / 612.5)
      → Recommend weighted loss with answer_weight=2.0

   2. Class distribution: ~40% NO, ~60% YES
      → Acceptable balance, no resampling needed

   3. Most vulnerabilities: 1-2 CWEs
      → Realistic distribution

   4. Few outliers with 2000+ tokens
      → Will be naturally truncated at max_seq_length=2560

🎯 OVERALL ASSESSMENT:
   Data quality: HIGH
   Format consistency: EXCELLENT
   Distribution: NATURAL
   Ready for training: YES ✅

   Both correlations are in healthy ranges:
   - Code vs Reasoning: 0.513 (moderate, shows natural scaling)
   - Reasoning vs Answer: 0.210 (weak, shows format consistency)

   No major data quality issues detected!

---

## Recommendation

# ✅ USE WEIGHTED LOSS!
# Answer tokens are only ~2% of total assistant tokens

```python
trainer = WeightedCoTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    reasoning_weight=1.0,
    answer_weight=2.0,  # 2x emphasis on the critical 2%
    answer_marker="Final Answer:",
)
```
