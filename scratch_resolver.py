import os

def resolve_file(filepath, choice_fn):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out = []
    chunk_head = []
    chunk_theirs = []
    state = 0
    
    for line in lines:
        if line.startswith('<<<<<<<'):
            state = 1
            chunk_head = []
            chunk_theirs = []
        elif line.startswith('======='):
            state = 2
        elif line.startswith('>>>>>>>'):
            resolved = choice_fn(filepath, chunk_head, chunk_theirs)
            out.extend(resolved)
            state = 0
        else:
            if state == 0:
                out.append(line)
            elif state == 1:
                chunk_head.append(line)
            elif state == 2:
                chunk_theirs.append(line)
                
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(out)

def resolver(filepath, head, theirs):
    if "ci-standard.yml" in filepath:
        # For CI, try to combine or prefer theirs
        if "Export Coverage" in "".join(theirs):
            return theirs
        elif "Configure Qt" in "".join(theirs):
            # Combined
            return [
                '          python -m ensurepip --upgrade\n',
                '          python -m pip install --upgrade pip setuptools\n',
                '          python -m pip install --ignore-installed --no-deps wheel\n',
                '          python -m pip install --force-reinstall --no-deps "sortedcontainers>=2.4.0" "hypothesis>=6.0.0"\n',
                '          python -m pip install -r requirements.txt\n',
                '          python -m pip install fastapi\n',
                '          python -m pip install httpx pytest-cov pytest-xdist pytest-benchmark\n',
                '\n',
                '      - name: Configure Qt for headless CI\n',
                '        run: |\n',
                '          # Use Qt offscreen platform — no X11/xcb infrastructure needed.\n',
                '          # PyQt6 widgets render off-screen; all Qt widget tests run safely.\n',
                '          echo "QT_QPA_PLATFORM=offscreen" >> $GITHUB_ENV\n'
            ]
        elif "ruff==" in "".join(head):
            return [
                '          python -m ensurepip --upgrade\n',
                '          python -m pip install --upgrade pip setuptools\n',
                '          python -m pip install --ignore-installed --no-deps wheel\n',
                '          python -m pip install --force-reinstall --no-deps "sortedcontainers>=2.4.0" "hypothesis>=6.0.0"\n',
                '          python -m pip install -r requirements.txt\n',
                '          python -m pip install fastapi\n',
                '          python -m pip install httpx pytest-cov pytest-xdist pytest-benchmark\n',
                '          python -m pip install ruff==0.14.10 bandit==1.7.7 pip-audit sortedcontainers types-PyYAML types-requests\n',
                '          python -m pip install --ignore-installed mypy==1.13.0\n'
            ]
        elif "grep -v -E" in "".join(theirs):
            return theirs
        return theirs
    
    if "limiter.py" in filepath:
        return theirs
        
    if "text_editor.py" in filepath:
        return theirs
        
    if "processor.py" in filepath:
        if "if not (path is not None):" in "".join(head):
            return head + theirs
        if "if not (df is not None):" in "".join(head):
            return head + theirs
        if "if not (start is not None):" in "".join(head):
            return head + theirs
        if "if not (target_rate is not None):" in "".join(head):
            return head + theirs
        return head
        
    if "safe_eval.py" in filepath:
        return head
        
    if "repository.py" in filepath:
        return theirs
        
    return head

for f in [
    "src/shared/python/data_processing/processor.py",
    "src/shared/python/model_generation/editor/text_editor.py",
    "src/shared/python/model_generation/library/repository.py",
    "src/shared/python/safe_eval.py",
    "src/web_applications/calculator/limiter.py",
    ".github/workflows/ci-standard.yml"
]:
    resolve_file(f, resolver)
