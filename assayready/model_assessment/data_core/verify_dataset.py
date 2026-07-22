import re
from pathlib import Path

# Strict regex guaranteeing uppercase ACGT exclusively
DNA_STRICT_RE = re.compile(rb"^[ACGT]+$")

def verify_fasta(filepath):
    print(f"\n[Verifying] {filepath.name} ({filepath.stat().st_size / 1e9:.2f} GB)")
    
    total_seqs = 0
    total_bases = 0
    failed_seqs = 0
    sampled = False
    
    with open(filepath, 'rb') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                total_seqs += 1
            else:
                total_bases += len(line)
                
                # Check strict filter
                if not DNA_STRICT_RE.match(line):
                    failed_seqs += 1
                    if failed_seqs <= 3:
                        print(f"  [ERROR] Found invalid sequence chunk: {line[:50]}...")
                
                # Print a small sample from the first sequence
                if not sampled:
                    print(f"  [SAMPLE] {line[:60].decode('utf-8')}...")
                    sampled = True
                    
    print(f"  Result: {total_seqs:,} sequences | {total_bases / 1e6:,.2f} MB")
    
    if failed_seqs > 0:
        print(f"  [FAIL] {failed_seqs:,} sequences contained invalid characters (lowercase, N, or non-ACGT).")
        return False
    print(f"  [PASS] 0% Ambiguity. 100% Strict ACGT.")
    return True

if __name__ == "__main__":
    data_dir = Path("data/domain_fastas")
    all_passed = True
    for fa_file in data_dir.glob("*.fa"):
        if "filtered" in fa_file.name: continue
        if not verify_fasta(fa_file):
            all_passed = False
            
    print("\n" + "="*40)
    if all_passed:
        print("FINAL RESULT: [SUCCESS] The entire dataset is perfectly clean.")
    else:
        print("FINAL RESULT: [FAILURE] Contaminants found in the dataset.")
    print("="*40)
