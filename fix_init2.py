import os
import ast

def add_all_to_init(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        src = f.read()
    if src.strip() == '': return
    
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets):
            return # Skip if already has __all__
            
    exports = []
    for node in tree.body:
        if hasattr(node, 'name'):
            if not node.name.startswith('_'):
                exports.append(node.name)
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                if not name.asname and not name.name.startswith('_'):
                    exports.append(name.name)
                elif name.asname and not name.asname.startswith('_'):
                    exports.append(name.asname)
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.asname and not name.asname.startswith('_'):
                    exports.append(name.asname)
                elif not name.name.startswith('_'):
                    exports.append(name.name.split('.')[0])
                    
    # Only if it has exports and is a 'public' looking file
    if exports and not 'tests' in filepath:
        all_stmt = '\n__all__ = [\n' + ''.join(f'    \"{e}\",\n' for e in sorted(list(set(exports)))) + ']\n'
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(all_stmt)
        print(f"Added __all__ to {filepath}")

for root, _, files in os.walk('src'):
    if '__init__.py' in files:
        add_all_to_init(os.path.join(root, '__init__.py'))
