#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Census of every `#[ignore]` attribute in the repo's tracked .rs files (GAP-408).

Usage: python scripts/ignore-census.py <repo-root> <ref> <out.tsv>
Reads blobs at <ref> (git ls-tree / git show), never the working tree. One row per attribute:
file, test fn, reason string (empty = bare), class. Class is a heuristic over path, reason and the
test body: cuda / vulkan / metal / weights-or-network / slow / other. Counts are ATTRIBUTES, not tests.
"""
import re,subprocess,sys,collections,csv
root=sys.argv[1]; ref=sys.argv[2]
files=subprocess.check_output(['git','-C',root,'ls-tree','-r','--name-only',ref],text=True).split('\n')
files=[f for f in files if f.endswith('.rs')]
rows=[]
for f in files:
    src=subprocess.check_output(['git','-C',root,'show',f'{ref}:./{f}'],text=True,errors='replace')
    if '#[ignore' not in src and '#[cfg_attr' not in src: continue
    L=src.split('\n')
    filecfg=' '.join(l for l in L[:40] if l.startswith('#![cfg') )
    for i,l in enumerate(L):
        m=re.match(r'\s*#\[ignore(?:\s*=\s*"(.*)"|\((?:reason\s*=\s*)?"(.*)"\))?\]',l)
        if not m: continue
        reason=(m.group(1) or m.group(2) or '')
        # find fn name and body
        name='?'; body=''
        for j in range(i+1,min(i+12,len(L))):
            mm=re.search(r'fn\s+(\w+)',L[j])
            if mm: name=mm.group(1); k=j; break
        else: k=i
        depth=0; seen=False; buf=[]
        for j in range(k,min(k+400,len(L))):
            buf.append(L[j]); depth+=L[j].count('{')-L[j].count('}')
            if '{' in L[j]: seen=True
            if seen and depth<=0: break
        body='\n'.join(buf)
        attrs=' '.join(L[max(0,i-4):i])
        rows.append((f,name,reason,filecfg+' '+attrs,body))
def cls(f,reason,cfg,body):
    s=(f+' '+reason+' '+cfg+' '+body).lower()
    if 'require_gpu_run_lock' in body or 'cudadevice' in s or 'feature = "cuda"' in s or 'cuda' in f.lower(): return 'cuda'
    if 'vulkan' in s: return 'vulkan'
    if 'metal' in s: return 'metal'
    if re.search(r'hf_|hub|download|network|http|huggingface|checkpoint|weights|\.gguf|safetensors|model_path|env::var',s): return 'weights-or-network'
    if re.search(r'slow|release|minutes|long|bench|stress|perf',s): return 'slow'
    return 'other'
out=[]
for f,n,r,c,b in rows:
    out.append((f,n,r,cls(f,r,c,b)))
w=csv.writer(open(sys.argv[3],'w',newline='',encoding='utf-8'),delimiter='\t',lineterminator='\n')
w.writerow(['file','test','reason','class'])
for o in sorted(out): w.writerow(o)
print(len(out),'attributes in',len(set(o[0] for o in out)),'files')
c=collections.Counter(o[3] for o in out); print(c)
print('bare',sum(1 for o in out if not o[2]),'with reason',sum(1 for o in out if o[2]))
