<!-- repo2kg-start -->
# Copilot Agent Instructions

This project has a code knowledge graph. Before exploring source files:

1. Read `CODEBASE.md` for a full project overview
2. Query `kg.toon` for specific code details:
   ```python
   import json; kg = json.load(open("kg.toon"))
   matches = [n for n in kg.values() if "KEYWORD" in n["name"].lower()]
   ```
3. Only read source files when you need implementation details beyond the 8-line preview
<!-- repo2kg-end -->
