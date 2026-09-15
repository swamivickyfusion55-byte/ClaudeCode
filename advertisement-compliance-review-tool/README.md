# Advertisement Compliance Review Tool — Architecture & Build Guide

`Advertisement_Compliance_Review_Tool.docx` is the deliverable: a solution
architecture and non-developer, step-by-step build guide for an
advertisement compliance review tool on Microsoft 365 (SharePoint + Power
Automate + Power Apps + Copilot).

## Files

- `Advertisement_Compliance_Review_Tool.docx` — the document itself.
- `architecture.png` / `architecture.svg` — the architecture diagram used in
  Section 3, generated from the SVG source.
- `build.js` — the script that generates the .docx (Node.js, using the
  `docx` npm package).

## Regenerating the document

If you need to edit the content and rebuild:

```bash
npm install docx sharp   # one-time, in this folder or any scratch folder
node build.js            # writes Advertisement_Compliance_Review_Tool.docx
```

To change the diagram, edit `architecture.svg`, then re-render it:

```bash
node -e "require('sharp')('architecture.svg', {density: 220}).png().toFile('architecture.png')"
```

then re-run `node build.js`.
