const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, Header, Footer, PageNumber, TableOfContents, LevelFormat,
  PageBreak, TabStopType, TableLayoutType, VerticalAlign, ExternalHyperlink
} = require("docx");

// ---------- constants ----------
const USABLE_WIDTH = 10080; // 8.5in page, 0.75in margins => 7in usable, in DXA
const NAVY = "1F2937";
const BLUE = "2563EB";
const TEAL = "0D9488";
const VIOLET = "7C3AED";
const AMBER = "D97706";
const GRAY = "6B7280";
const LGRAY = "F3F4F6";
const BORDER_GRAY = "D1D5DB";

// ---------- numbering ----------
const numberingConfigs = [
  {
    reference: "bullets",
    levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.START,
        style: { paragraph: { indent: { left: 450, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.START,
        style: { paragraph: { indent: { left: 810, hanging: 270 } } } },
    ],
  },
];
let stepCounter = 0;
function newStepsRef() {
  stepCounter += 1;
  const reference = "steps" + stepCounter;
  numberingConfigs.push({
    reference,
    levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
        style: { paragraph: { indent: { left: 470, hanging: 340 } } } },
    ],
  });
  return reference;
}

// ---------- run/paragraph helpers ----------
function runsFromSpec(spec) {
  if (typeof spec === "string") return [new TextRun({ text: spec })];
  return spec.map((r) => {
    if (typeof r === "string") return new TextRun({ text: r });
    return new TextRun({
      text: r.text,
      bold: !!r.bold,
      italics: !!r.italic,
      color: r.color,
      font: r.mono ? "Consolas" : undefined,
      size: r.mono ? 19 : undefined,
    });
  });
}

function p(spec, opts = {}) {
  return new Paragraph({
    children: runsFromSpec(spec),
    spacing: { after: opts.after ?? 160, before: opts.before ?? 0, line: opts.line },
    alignment: opts.alignment,
  });
}

function h1(text, n) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text: (n ? n + ". " : "") + text })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text })],
  });
}

function bullets(items) {
  return items.map(
    (spec) =>
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: runsFromSpec(spec),
        spacing: { after: 90 },
      })
  );
}

function steps(items) {
  const ref = newStepsRef();
  return items.map(
    (spec) =>
      new Paragraph({
        numbering: { reference: ref, level: 0 },
        children: runsFromSpec(spec),
        spacing: { after: 110 },
      })
  );
}

function callout(label, spec, color = BLUE) {
  const runs = [new TextRun({ text: label + "  ", bold: true, color })].concat(runsFromSpec(spec));
  return new Paragraph({
    shading: { fill: LGRAY, type: ShadingType.CLEAR },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GRAY, space: 6 },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GRAY, space: 6 },
      left: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GRAY, space: 8 },
      right: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GRAY, space: 8 },
    },
    spacing: { before: 120, after: 200 },
    children: runs,
  });
}

function formulaBox(label, code) {
  return new Paragraph({
    shading: { fill: "111827", type: ShadingType.CLEAR },
    border: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "111827", space: 8 },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "111827", space: 8 },
      left: { style: BorderStyle.SINGLE, size: 2, color: "111827", space: 10 },
      right: { style: BorderStyle.SINGLE, size: 2, color: "111827", space: 10 },
    },
    spacing: { before: 100, after: 200 },
    children: [
      new TextRun({ text: label, color: "9CA3AF", size: 18, break: 0 }),
      new TextRun({ text: code, color: "F9FAFB", font: "Consolas", size: 19, break: 1 }),
    ],
  });
}

function spacer(size = 120) {
  return new Paragraph({ spacing: { after: size }, children: [] });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ---------- table helper ----------
function cell(text, opts = {}) {
  const paragraphs = Array.isArray(text)
    ? text.map((t) => (t instanceof Paragraph ? t : p(t, { after: 60 })))
    : [typeof text === "string" ? p(text, { after: 0 }) : text];
  return new TableCell({
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.header ? { fill: NAVY, type: ShadingType.CLEAR } : opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 90, bottom: 90, left: 120, right: 120 },
    children: opts.header
      ? paragraphs.map((par) => new Paragraph({ children: [new TextRun({ text: par.root ? "" : "", bold: true })], spacing: { after: 0 } }))
      : paragraphs,
  });
}

// simpler header-cell builder (text only, bold white)
function headCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { fill: NAVY, type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 20 })], spacing: { after: 0 } })],
  });
}
function bodyCell(content, width, shadeFill) {
  const paragraphs = Array.isArray(content) ? content : [content];
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shadeFill ? { fill: shadeFill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 90, bottom: 90, left: 120, right: 120 },
    children: paragraphs.map((t) =>
      typeof t === "string"
        ? new Paragraph({ children: [new TextRun({ text: t, size: 20 })], spacing: { after: 40 } })
        : t
    ),
  });
}

function dataTable(headers, widths, rows, opts = {}) {
  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((htext, i) => headCell(htext, widths[i])),
  });
  const bodyRows = rows.map(
    (r, ri) =>
      new TableRow({
        cantSplit: true,
        children: r.map((c, i) => bodyCell(c, widths[i], opts.zebra && ri % 2 === 1 ? "F9FAFB" : undefined)),
      })
  );
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    rows: [headerRow, ...bodyRows],
  });
}

function imageFigure(imgPath, widthPx, heightPx, caption) {
  const data = fs.readFileSync(imgPath);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [
        new ImageRun({
          type: "png",
          data,
          transformation: { width: widthPx, height: heightPx },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: GRAY })],
    }),
  ];
}

// ---------- block renderer (for phases) ----------
function renderBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    if (b.type === "p") out.push(p(b.text, { after: b.after ?? 140 }));
    else if (b.type === "steps") out.push(...steps(b.items));
    else if (b.type === "bullets") out.push(...bullets(b.items));
    else if (b.type === "callout") out.push(callout(b.label, b.text, b.color));
    else if (b.type === "formula") out.push(formulaBox(b.label, b.code));
    else if (b.type === "table") out.push(dataTable(b.headers, b.widths, b.rows, b.opts || {}), spacer(200));
    else if (b.type === "h3") out.push(h3(b.text));
    else if (b.type === "spacer") out.push(spacer(b.size));
  }
  return out;
}

// =====================================================================
// CONTENT DATA
// =====================================================================

const appsRows = [
  ["Site, register &\ndashboard", "SharePoint Online", "Hosts the site, the Home/Dashboard page, the Advertisement Log, Checklist Master, Checklist Responses and the Regulations Library.", "Included in essentially every Microsoft 365 Business or Enterprise plan."],
  ["Automation", "Power Automate\n(cloud flows, standard connectors only)", "Assigns the unique Advertisement ID, builds each ad's checklist, links Page 1 to Page 2, sends notifications, and applies CCO-approved checklist updates.", "This design only uses standard SharePoint / Outlook / Teams / Approvals connectors, which are included with Microsoft 365 — no separate Power Automate license is required."],
  ["Nicer forms (optional upgrade)", "Power Apps\n(canvas app on top of your SharePoint lists)", "A friendlier single-screen “New Ad Review” form and a true single-page dynamic checklist for Page 2, instead of plain list forms.", "Covered by the Power Apps use rights included with most Microsoft 365 plans, as long as you only connect to SharePoint (no Dataverse or premium connectors)."],
  ["Checklist refresh assistant", "Microsoft 365 Copilot chat and/or a Copilot Studio agent", "Reads uploaded regulations and the current Checklist Master, then drafts suggested checklist updates for the CCO to review.", "Usually a paid add-on. Confirm with your Microsoft 365 admin exactly what is already licensed in your tenant before committing to the Tier 3 build in Phase 7."],
  ["Dashboard charts (optional)", "SharePoint “Quick Chart” web part (or Power BI)", "Simple visual counts — ads by category, by decision, by status — on the Home/Dashboard page.", "Quick Chart is built into SharePoint at no extra cost. Power BI is optional and typically a separate license."],
  ["Setup & administration", "Microsoft 365 admin center / SharePoint admin center", "Create the site, confirm the 3 users' accounts, and check license availability before you start.", "Included; you (or whoever administers Microsoft 365 for your organization) needs permission to create SharePoint sites."],
];

const dataModelSections = [
  {
    name: "1. Ad Categories",
    kind: "List",
    purpose: "The list of advertisement types your team reviews against (e.g., Digital & Social Media, Print, Website). Everything else keys off this list, so build it first.",
    rows: [
      ["Title", "Single line of text", "The category name, e.g. “Digital & Social Media”. This is the list's built-in first column — just rename it if you like."],
      ["Description", "Multiple lines of text", "A short definition/examples shown to reviewers so category choice is consistent."],
      ["Active", "Yes/No (choice)", "Only categories marked Active appear as a choice when someone logs a new advertisement."],
    ],
  },
  {
    name: "2. Checklist Master",
    kind: "List",
    purpose: "Your backend checklist master — one row per checklist question per category. This is what Copilot helps you keep current in Phase 7.",
    rows: [
      ["Title", "Single line of text", "A short name for the checklist item, e.g. “Required disclosures visible”."],
      ["Category", "Lookup → Ad Categories", "Which advertisement category this question applies to."],
      ["Item Number", "Number", "Controls the display order of questions within a category."],
      ["Checklist Question", "Multiple lines of text", "The actual requirement or question a reviewer must answer."],
      ["Regulatory Reference", "Single line of text", "The rule, policy clause or regulation this question is based on."],
      ["Guidance Notes", "Multiple lines of text", "Extra explanation or examples to help a reviewer answer consistently."],
      ["Mandatory", "Choice: Yes / No", "Whether a “Non-Compliant” answer here should block approval."],
      ["Version", "Single line of text", "e.g. “v1.3” — increase this each time the wording changes."],
      ["Effective Date", "Date", "When this version of the question became applicable."],
      ["Status", "Choice: Active / Retired", "Only Active items are pulled into new checklists."],
    ],
  },
  {
    name: "3. Checklist Master Archive",
    kind: "List",
    purpose: "A permanent record of every retired checklist item, so you can always see what a question used to say. Create it with the same columns as Checklist Master (copy/paste the column list above), plus two extra columns:",
    rows: [
      ["Archived Date", "Date", "When this version was superseded."],
      ["Superseded By", "Single line of text", "The Title/Item Number of the new Checklist Master row that replaced it."],
    ],
  },
  {
    name: "4. Advertisement Log",
    kind: "List",
    purpose: "The master register — “Page 1”. One row per advertisement reviewed. This is what the CCO dashboard reads from.",
    rows: [
      ["Title", "Single line of text", "The advertisement's name or short description."],
      ["AdvertisementID", "Single line of text (indexed)", "The auto-generated unique number, e.g. ADV-2026-0001. Name this column with no space — see the tip in Phase 2."],
      ["Category", "Lookup → Ad Categories", "Drives which checklist gets attached to this advertisement (the “type” in your log)."],
      ["Description", "Multiple lines of text", "What the ad is, the channel, the campaign, etc."],
      ["Submitted By", "Person", "Auto-filled with whoever created the item."],
      ["Date Submitted", "Date", "Auto-filled at creation."],
      ["Reviewed By", "Person", "The assigned reviewer."],
      ["Review Summary", "Multiple lines of text", "A plain-English summary of the review outcome, for the log and dashboard."],
      ["Checklist Link", "Hyperlink", "Auto-built link straight to this advertisement's filled-in checklist (“Page 2”)."],
      ["Compliance Comments", "Multiple lines of text", "The reviewer's compliance narrative / rationale."],
      ["Decision", "Choice: Approved / Approved with Conditions / Rejected / Pending Review / Escalated", "The formal decision."],
      ["Decision Date", "Date", "When the decision was recorded."],
      ["Status", "Choice: Draft / In Review / Completed", "Workflow status shown on the dashboard."],
    ],
  },
  {
    name: "5. Checklist Responses",
    kind: "List",
    purpose: "The filled-in checklist — “Page 2”. One row per checklist question per advertisement. Question wording is copied (not just linked) from Checklist Master at creation time, so a later checklist update never silently rewrites what an old advertisement was actually reviewed against.",
    rows: [
      ["Title", "Single line of text", "Auto-built, e.g. “ADV-2026-0001 – Item 3”."],
      ["AdvertisementID", "Single line of text (indexed)", "Matches the Advertisement Log's AdvertisementID. Kept as plain text (not a lookup) so the Page 2 link/filter in Phase 4 stays simple. Same no-space naming tip applies."],
      ["Checklist Item (source)", "Lookup → Checklist Master", "Traceability back to the master question that was used."],
      ["Checklist Question (snapshot)", "Multiple lines of text", "Copied from Checklist Master when the checklist was built — freezes the exact wording reviewed against."],
      ["Regulatory Reference (snapshot)", "Single line of text", "Copied at the same time, for the same reason."],
      ["Checklist Version (snapshot)", "Single line of text", "Which version of the question this review used."],
      ["Response", "Choice: Compliant / Non-Compliant / Not Applicable / Pending", "The reviewer's answer to this item."],
      ["Comments", "Multiple lines of text", "Explanation or evidence notes for this specific answer."],
      ["Evidence", "Attachment", "Screenshot or supporting file, if needed."],
      ["Reviewed By", "Person", "Who answered this item."],
      ["Review Date", "Date", "When it was answered."],
    ],
  },
  {
    name: "6. Regulations Library",
    kind: "Document Library",
    purpose: "Where you file the actual regulatory documents — this is what the Copilot interface in Phase 7 reads from. Create it as a Document Library (not a list), then add these columns to it.",
    rows: [
      ["Regulator", "Single line of text", "e.g. the name of the regulatory body or internal policy owner."],
      ["Regulation Name", "Single line of text", "Title or number of the regulation/circular."],
      ["Effective Date", "Date", "When the regulation takes (or took) effect."],
      ["Related Category", "Lookup → Ad Categories (optional)", "Which advertisement category it mainly affects."],
      ["Summary", "Multiple lines of text", "A short plain-English summary — Copilot can help draft this."],
    ],
  },
];

const phases = [
  {
    heading: "Phase 0 — Before You Start",
    intro: "A short planning pass now saves rework later.",
    blocks: [
      { type: "steps", items: [
        "Confirm with your Microsoft 365 administrator that you (the CCO) can create a new SharePoint site. If not, ask them to create one and make you the Owner.",
        "Confirm what is already licensed in your tenant — Microsoft 365 with SharePoint is almost certainly there; separately check whether anyone already has Microsoft 365 Copilot or Copilot Studio, since that decides which option you use in Phase 7.",
        "Decide your Advertisement ID format. Recommended: ADV-YYYY-#### (e.g., ADV-2026-0001), restarting the sequence each calendar year — see Appendix A.",
        "Gather your current compliance checklists (Excel, Word, or paper) for each advertisement category into one place, ideally one Excel file with one tab per category — you'll paste this into Checklist Master in Phase 3.",
        "Confirm the Microsoft 365 sign-in email addresses of your 2 team members — you'll add them as site Members in Phase 1.",
      ]},
    ],
  },
  {
    heading: "Phase 1 — Create the SharePoint Site",
    blocks: [
      { type: "steps", items: [
        "Go to your organization's SharePoint start page (the app launcher / waffle icon in Outlook, Teams or office.com → SharePoint).",
        "Click “+ Create site”.",
        [{text:"Choose "},{text:"Team site",bold:true},{text:" (recommended for a 3-person team — simplest permissions). Name it, e.g. “Advertisement Compliance Review”."}],
        "On the “Add members” step, add your 2 teammates. You remain the site Owner by default as its creator.",
        "Once created, go to the gear icon → Site Permissions and confirm: you = Owner, both teammates = Members. There should be no other members yet.",
      ]},
      { type: "callout", label: "Tip:", text: "Also turn off external sharing while you're there: Site Permissions → Site sharing settings → set to “Only people in your organization”. This is an internal compliance workpaper — it should never be shareable outside your organization by accident." },
    ],
  },
  {
    heading: "Phase 2 — Build the SharePoint Lists",
    intro: "You're about to create six things — five lists and one document library — using the column definitions in Section 5. The same two steps apply every time.",
    blocks: [
      { type: "h3", text: "How to create any list" },
      { type: "steps", items: [
        "From the site, click “+ New” → “List” → “Blank list” (for the Regulations Library, choose “Document library” instead).",
        "Name it exactly as shown in Section 5 and click Create.",
        [{text:"For every row in that list's column table, click "},{text:"+ Add column",bold:true},{text:", pick the matching type, type the column name exactly as shown, and Save. Repeat for every row."}],
      ]},
      { type: "callout", label: "Tip — naming columns:", text: "Wherever a column name is written with no space, such as AdvertisementID, type it exactly that way (no space) when you create it. SharePoint bakes spaces into a hidden internal name that makes the hyperlink formula in Phase 4 much harder to get right. Every other column is fine with normal spacing." },
      { type: "p", text: "Build them in this order, because later lists look up to earlier ones:" },
      { type: "bullets", items: [
        "Ad Categories",
        "Checklist Master (looks up to Ad Categories)",
        "Checklist Master Archive (same columns as Checklist Master, plus the two extra ones)",
        "Advertisement Log (looks up to Ad Categories)",
        "Checklist Responses (looks up to Checklist Master)",
        "Regulations Library (a document library, with its own metadata columns)",
      ]},
      { type: "h3", text: "Two settings to switch on afterwards" },
      { type: "steps", items: [
        [{text:"On "},{text:"Advertisement Log",bold:true},{text:", turn on versioning: List settings → Versioning settings → “Create a version each time you edit an item” = Yes. Do the same on Checklist Master."}],
        [{text:"Once Phase 4's flow is built and tested, come back and set "},{text:"AdvertisementID",bold:true},{text:" on Advertisement Log to “Enforce unique values” (column settings → Edit column) — this guarantees two ads can never end up with the same number. Doing this "},{text:"before",italic:true},{text:" the flow exists can make testing confusing, so leave it until after Phase 4."}],
      ]},
    ],
  },
  {
    heading: "Phase 3 — Seed Your Checklist Master",
    blocks: [
      { type: "steps", items: [
        "Add your categories to Ad Categories first — Appendix B has a starter list you can copy in and edit.",
        [{text:"Open Checklist Master, switch to "},{text:"grid view",bold:true},{text:" (View options → Edit in grid view, top-right of the list) — this behaves like an Excel sheet and lets you paste many rows at once straight from your gathered checklist content."}],
        "Paste in your checklist questions, category by category. Set Status = Active on everything that should be live immediately, and give everything Version = “v1.0” and today's date as the Effective Date.",
      ]},
      { type: "callout", label: "Note:", text: "Don't have your checklist content digitized yet? Add 3–5 items per category by hand to get the tool working end-to-end, then come back and bulk-paste the rest once it's typed up — nothing else in this guide depends on the list being complete." },
    ],
  },
  {
    heading: "Phase 4 — Build the Power Automate Flows",
    intro: "All five flows below are built in the same place — make.powerautomate.com → My flows → + New flow — and all of them use only standard SharePoint / Outlook / Teams / Approvals connectors, so no extra Power Automate license is needed.",
    blocks: [
      { type: "h3", text: "Flow A — New Advertisement Setup" },
      { type: "p", text: "This is the flow that does most of the work: it assigns the ID, builds the checklist, and creates the Page 2 link, all from one trigger." },
      { type: "steps", items: [
        [{text:"Trigger: "},{text:"When an item is created",bold:true},{text:" (SharePoint) — site: your site, list: Advertisement Log."}],
        [{text:"Add "},{text:"Update item",bold:true},{text:" (same list, Id = ID from the trigger). Set the AdvertisementID field using this expression (Insert dynamic content → Expression tab):"}],
      ]},
      { type: "formula", label: "Paste into the AdvertisementID field:", code: "concat('ADV-', formatDateTime(utcNow(),'yyyy'), '-', formatNumber(triggerOutputs()?['body/ID'],'0000'))" },
      { type: "p", text: "This turns SharePoint's own built-in item number into ADV-2026-0001, ADV-2026-0002, and so on — always unique, even if two people submit at the same moment, because it rides on a number SharePoint itself never repeats." },
      { type: "steps", items: [
        [{text:"Add "},{text:"Get items",bold:true},{text:" from Checklist Master. Click “Add” next to Filter Query and use the built-in filter picker (no typing needed) for: Category equals the trigger's Category, and Status equals Active."}],
        [{text:"Add "},{text:"Apply to each",bold:true},{text:" over those results. Inside the loop, add "},{text:"Create item",bold:true},{text:" in Checklist Responses: set AdvertisementID to the trigger's AdvertisementID text, Checklist Item to the current loop item, and copy across Checklist Question, Regulatory Reference and Version from the loop item into the matching “(snapshot)” columns. Set Response = “Pending”."}],
        [{text:"After the loop, add another "},{text:"Update item",bold:true},{text:" on Advertisement Log to set Checklist Link:"}],
      ]},
      { type: "formula", label: "Address field (edit the site/list part for your tenant):", code: "concat('https://yourtenant.sharepoint.com/sites/AdComplianceReview/Lists/Checklist Responses/AllItems.aspx?FilterField1=AdvertisementID&FilterValue1=', outputs('Update_item')?['body/AdvertisementID'])" },
      { type: "callout", label: "If the filtered link doesn't work:", text: "This URL-filter trick relies on the list's classic-style view. If your Checklist Responses view has been switched to a newer format and the filter doesn't apply automatically, either switch that view back to the default format, or skip straight to the optional Power Apps upgrade in Phase 6, which replaces this link with a real single-page checklist screen.", color: AMBER },
      { type: "p", text: "Save Flow A, then test it: add one item to Advertisement Log with a category selected, and confirm the AdvertisementID appears, matching checklist rows appear in Checklist Responses, and the Checklist Link opens a view showing just those rows." },
      { type: "h3", text: "Flow B — Notify Reviewer on Submission" },
      { type: "steps", items: [
        "Trigger: When an item is created (Advertisement Log).",
        "Action: Send an email (Outlook) or post a message in Teams to the assigned Reviewed By person, with the ad title, category and a link to the item.",
      ]},
      { type: "h3", text: "Flow C — Notify on Decision" },
      { type: "steps", items: [
        "Trigger: When an item is modified (Advertisement Log). In the trigger's settings, add a trigger condition so it only fires when the Decision field is not blank.",
        "Action: Send an email to Submitted By with the Decision and Compliance Comments, and set Status = “Completed”.",
      ]},
      { type: "h3", text: "Flow D — Overdue Review Reminder" },
      { type: "steps", items: [
        "Trigger: Recurrence — once a day, e.g. 8:00 AM.",
        "Action: Get items from Advertisement Log where Status equals “In Review” and Date Submitted is older than your SLA (e.g. 5 days) → Apply to each → send a reminder email to Reviewed By.",
      ]},
      { type: "h3", text: "Flow E — Checklist Master Update" },
      { type: "p", text: "This is the flow the CCO (and later, Copilot) uses to actually change the checklist master — with a human approval step built in, so nothing reaches the live checklist unreviewed." },
      { type: "steps", items: [
        [{text:"Trigger: "},{text:"Manually trigger a flow",bold:true},{text:" (Instant cloud flow), with input fields: Category, Checklist Question, Regulatory Reference, Guidance Notes, Mandatory, New Version, Effective Date, and an optional “Item to Replace” (the Title/Item Number of an existing Checklist Master row being superseded)."}],
        [{text:"Add "},{text:"Start and wait for an approval",bold:true},{text:", assigned to you (the CCO), showing the submitted details. Nothing below this step runs until you approve it."}],
        "If “Item to Replace” was provided: Get that item from Checklist Master → Create a matching item in Checklist Master Archive (copy all its fields, plus today as Archived Date) → Update the original item's Status to “Retired”.",
        "Create the new item in Checklist Master from the trigger inputs, with Status = “Active”.",
      ]},
      { type: "callout", label: "Why the approval step matters:", text: "Flow E is the one place regulatory judgement turns into a live checklist requirement. Keeping a mandatory human approval here means that even once Copilot Studio can call this flow directly in Phase 7, an AI suggestion can never become an official compliance criterion without the CCO explicitly signing off." },
    ],
  },
  {
    heading: "Phase 5 — Build the Pages",
    blocks: [
      { type: "h3", text: "5.1 — Home & Dashboard Page" },
      { type: "steps", items: [
        "Edit the site's Home page (Edit, top right).",
        "Add a List web part pointed at Advertisement Log, using a view grouped by Status, showing AdvertisementID, Category, Reviewed By, Decision and Checklist Link.",
        "Add a Quick Chart web part summarizing Advertisement Log by Category and another by Decision.",
        "Add a Quick links web part with buttons to the New Ad Review and Checklist Admin pages (built next).",
      ]},
      { type: "h3", text: "5.2 — New Ad Review Page (Page 1)" },
      { type: "p", text: "For the initial build, the simplest approach is a page with a button straight to the list's own New Item form — no separate page content needed:" },
      { type: "steps", items: [
        "Create a Site Page named “New Ad Review”.",
        "Add a Button web part labelled “+ Submit New Advertisement for Review”, linking to the Advertisement Log list's New Item form (open the list, click + New once to see its URL, ending in NewForm.aspx, and paste that).",
      ]},
      { type: "h3", text: "5.3 — Checklist Detail (Page 2)" },
      { type: "p", text: "In the MVP, “Page 2” is simply the filtered Checklist Responses view that Flow A links to — you don't need to build a separate SharePoint page for it. Reviewers click Checklist Link on the log and land directly on that advertisement's checklist, where they can open each row to answer it." },
      { type: "h3", text: "5.4 — Checklist Admin Page" },
      { type: "steps", items: [
        "Create a Site Page named “Checklist Admin”.",
        "Add a List web part pointed at Checklist Master, grouped by Category.",
        "Add a button/link to run Flow E (Power Automate can generate a shareable link for a manually-triggered flow — open the flow → “Get a link”).",
        "You'll add the Copilot interface to this page in Phase 7.",
      ]},
      { type: "p", text: "Finally, add all of these pages to the site's top navigation so the 3 of you can get anywhere in one click." },
    ],
  },
  {
    heading: "Phase 6 (Optional) — Upgrade to Power Apps",
    intro: "Everything in this guide works without this phase. Do it later if the plain SharePoint forms and filtered view feel clunky and you want a more polished experience.",
    blocks: [
      { type: "steps", items: [
        [{text:"Quick win: open Advertisement Log → Integrate → Power Apps → "},{text:"Customize forms",bold:true},{text:". This opens Power Apps Studio already wired to your list. Rearrange fields, Save, Publish — it becomes the new New Item form automatically, no data connections to configure."}],
        "Bigger upgrade, for a real single-page Page 2: go to make.powerapps.com → + Create → Blank canvas app. Connect it to Advertisement Log and Checklist Responses. Add a Gallery bound to Checklist Responses, filtered to AdvertisementID equal to a URL parameter (Param(\"AdID\")). Add a dropdown for Response and a text box for Comments inside the gallery, saved back with Patch().",
        "Embed the finished app on the Checklist Detail page using the Power Apps web part.",
      ]},
      { type: "callout", label: "Honest assessment:", text: "The Gallery-and-Patch() step above is the one genuinely “builder” task in this whole guide. If it feels like too much, skip it — the filtered list view from Phase 5.3 does the same job, just with one extra click to open each item." },
    ],
  },
  {
    heading: "Phase 7 — Set Up the Copilot Checklist-Refresh Interface",
    intro: "Build this in tiers. Tier 1 always works and costs nothing; only go further if your organization already has the relevant Copilot license.",
    blocks: [
      { type: "h3", text: "Tier 1 — Manual (no license needed, your fallback either way)" },
      { type: "p", text: "Whenever a new regulation or amendment appears, upload it to the Regulations Library, then compare it to the relevant Checklist Master items yourself and update them through Flow E. This works immediately and should remain your fallback regardless of which tier below you also build." },
      { type: "h3", text: "Tier 2 — Microsoft 365 Copilot chat (if already licensed)" },
      { type: "p", text: "Open Copilot in Teams, Word or the Microsoft 365 app, reference the uploaded regulation file and the Checklist Master list, and ask it to summarize what changed and suggest which checklist rows need new or updated wording. Copy its suggestions into Flow E yourself — no build required, just a way of working." },
      { type: "h3", text: "Tier 3 — A dedicated Copilot Studio agent (if you have Copilot Studio access)" },
      { type: "steps", items: [
        "Go to copilotstudio.microsoft.com → + Create → New agent. Name it, e.g., “Ad Compliance Checklist Assistant”.",
        "Give it instructions in plain English, along these lines:",
      ]},
      { type: "callout", label: "Suggested agent instructions:", text: "You help a compliance team keep their advertisement review checklist current. When asked about a regulation, read it from the Regulations Library. Compare it to the current Checklist Master for the relevant category. Propose specific additions, edits or retirements, each with its regulatory citation. Never claim you have updated anything — always end by asking the compliance officer to confirm before any change is applied." },
      { type: "steps", items: [
        "Add Knowledge sources: your SharePoint site → select the Regulations Library and the Checklist Master list.",
        "Optional, more advanced: add an Action pointing at Flow E (“Checklist Master Update”) as a tool the agent can call, only after you've typed a clear confirmation in the chat. Flow E's own approval step (Phase 4) still has the final word.",
        "Test it in the built-in test pane, e.g.: “What changed in [regulation name] and how should the Print checklist change?”",
        "Publish the agent, then add it to Teams (easiest for daily use) and/or embed or link it on the Checklist Admin page from Phase 5.4.",
      ]},
    ],
  },
  {
    heading: "Phase 8 — Permissions & Security Check",
    blocks: [
      { type: "bullets", items: [
        "Site membership is exactly 3 people, with the right roles (you = Owner, 2 teammates = Members).",
        "Versioning is on for Advertisement Log, Checklist Master and Checklist Master Archive.",
        "External sharing is off (Phase 1).",
        [{text:"If your organization uses Microsoft Purview retention policies, ask your records/IT team whether Advertisement Log needs a retention label — compliance workpapers often carry a minimum retention period. That's a policy decision for your records owner, not something to decide unilaterally while building the tool."}],
      ]},
    ],
  },
  {
    heading: "Phase 9 — Test Before You Rely On It",
    blocks: [
      { type: "table", widths: [700, 4550, 4830], headers: ["#", "Test", "Expected result"], rows: [
        ["1", "Add a new ad in category “Print”.", "AdvertisementID auto-fills, e.g. ADV-2026-0001; matching Print checklist items appear in Checklist Responses."],
        ["2", "Open the Checklist Link from the log.", "The filtered view shows only that ad's checklist items — nothing from any other advertisement."],
        ["3", "Answer all checklist items, then set a Decision on the log item.", "Decision Date fills in, Status becomes Completed, and the submitter gets a notification (Flow C)."],
        ["4", "Submit a second ad the same day.", "It gets a different, sequential AdvertisementID — no collision with test 1."],
        ["5", "Run Flow E to replace a checklist item.", "An approval request reaches you; only after you approve does Checklist Master show the new item, with the old one moved to Archive and marked Retired."],
        ["6", "Open the Home/Dashboard page.", "Both test ads appear with correct counts and status."],
      ]},
      { type: "p", text: "Finally, walk your 2 team members through logging a new ad, completing a checklist, and reading the dashboard — 30–45 minutes is usually enough." },
    ],
  },
  {
    heading: "Phase 10 — Keep It Running",
    blocks: [
      { type: "bullets", items: [
        "Set a recurring monthly or quarterly reminder to check for new regulatory guidance and run the Phase 7 refresh process.",
        "Periodically export Advertisement Log and Checklist Master to Excel as an offline backup (list → Export to Excel), in addition to SharePoint's own version history.",
        "Re-run the Phase 9 checklist any time you change a flow or restructure the checklist.",
      ]},
    ],
  },
];

const timelineRows = [
  ["0", "Prerequisites & planning", "2–3 hours", "Easy"],
  ["1", "Create the SharePoint site", "30 minutes", "Easy"],
  ["2", "Build the 6 lists/library", "3–4 hours", "Easy, repetitive"],
  ["3", "Seed the Checklist Master", "4–8 hours*", "Easy, time-consuming"],
  ["4", "Build the 5 Power Automate flows", "1–2 days", "Moderate"],
  ["5", "Build the 4 pages", "3–4 hours", "Easy"],
  ["6", "Power Apps upgrade (optional)", "1–2 days", "Moderate–High"],
  ["7", "Copilot interface", "Tier 1: none · Tier 2: ~1 hour · Tier 3: ~1 day", "Easy–Moderate"],
  ["8", "Permissions & security check", "1 hour", "Easy"],
  ["9", "Testing & training", "3–4 hours", "Easy"],
  ["10", "Ongoing maintenance", "~1 hour / month", "Easy"],
];

const glossaryRows = [
  ["SharePoint site", "The web space that holds your pages, lists and document library — the front end in this design."],
  ["List", "A SharePoint table, like a shared Excel sheet with structured columns — used here as the tool's database."],
  ["Document library", "Like a list, but built for storing files (here: the Regulations Library)."],
  ["Lookup column", "A column that points to a row in another list, keeping the two connected (e.g., Checklist Master → Ad Categories)."],
  ["Power Automate / cloud flow", "Microsoft's no-code automation tool. A “flow” is one automated recipe: a trigger plus a series of actions."],
  ["Trigger / action / connector", "A trigger starts a flow (e.g., “when an item is created”); actions are the steps it then runs; a connector is the service an action talks to (e.g., SharePoint, Outlook). “Standard” connectors are included with Microsoft 365; “premium” connectors usually cost extra — this design avoids premium connectors entirely."],
  ["Power Apps / canvas app", "Microsoft's no-code app builder. A “canvas app” is a custom screen (or set of screens) you lay out yourself, as used in the optional Phase 6 upgrade."],
  ["Copilot / Copilot Studio / agent", "Copilot is Microsoft's AI assistant; Copilot Studio is the tool for building a custom, purpose-built version of it (an “agent”) grounded on your own content — here, the Regulations Library and Checklist Master."],
  ["OData filter query", "The technical name for the “filter” syntax SharePoint and Power Automate use behind the scenes; the filter-builder UI in Phase 4 writes this for you."],
  ["Indexed column", "A column marked for fast filtering/search — recommended for AdvertisementID so lookups stay quick as the log grows."],
  ["Versioning", "SharePoint's built-in change history for list items — the source of your audit trail."],
];

const appendixBRows = [
  ["Digital & Social Media", "Does the advertisement include all legally required disclosures/disclaimers, visibly and legibly?", "[Insert your applicable rule]"],
  ["Print", "Are all performance claims or statistics substantiated by a verifiable source on file?", "[Insert your applicable rule]"],
  ["Email / Direct Marketing", "Does the communication include a compliant opt-out / unsubscribe mechanism?", "[Insert your applicable rule]"],
  ["Website", "Are hyperlinked third-party sites and related disclaimers appropriately flagged?", "[Insert your applicable rule]"],
  ["Sales Collateral / Brochures", "Is the material consistent with previously approved product/service descriptions?", "[Insert your applicable rule]"],
  ["Sponsorship / Event Materials", "Are sponsor/organizer obligations and required disclosures represented accurately?", "[Insert your applicable rule]"],
];

const appendixDRows = [
  ["Create/manage the SharePoint site", "https://[yourtenant].sharepoint.com, or office.com → app launcher → SharePoint"],
  ["Build/manage Power Automate flows", "https://make.powerautomate.com"],
  ["Build/manage Power Apps", "https://make.powerapps.com"],
  ["Build a Copilot Studio agent", "https://copilotstudio.microsoft.com"],
  ["Manage users & licenses", "https://admin.microsoft.com"],
];

// =====================================================================
// DOCUMENT ASSEMBLY
// =====================================================================

const children = [];

// ---- cover page ----
children.push(
  new Paragraph({ spacing: { before: 1800, after: 0 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text: "ADVERTISEMENT COMPLIANCE REVIEW TOOL", bold: true, size: 52, color: NAVY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 20 } },
    children: [new TextRun({ text: "Solution Architecture & Step-by-Step Build Guide", size: 30, color: BLUE })],
  }),
  new Paragraph({ spacing: { after: 900 }, children: [] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Prepared for: Compliance Function — CCO and Review Team (3 users)", size: 22, color: "374151" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Platform: Microsoft 365 — SharePoint, Power Automate, Power Apps, Copilot", size: 22, color: "374151" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Date: September 15, 2026", size: 22, color: "374151" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Classification: Internal — Confidential", size: 20, color: GRAY, italics: true })],
  }),
  pageBreak()
);

// ---- TOC ----
children.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "Table of Contents" })] }),
  p([{ text: "This page number list fills in automatically once opened in Word: right-click anywhere below and choose “Update Field” (or select this page and press F9).", italic: true, color: GRAY }], { after: 200 }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  pageBreak()
);

// ---- 1. Executive Summary ----
children.push(h1("Executive Summary", 1));
children.push(p(
  "This document sets out a simple, no-custom-code way to build an advertisement compliance review tool for your team — the CCO plus 2 reviewers — entirely on tools your organization already licenses through Microsoft 365: SharePoint for the front end and system of record, Power Automate for the logic, and Copilot for keeping the checklist current with regulatory change."
));
children.push(p("The tool will let your team:", { after: 100 }));
children.push(...bullets([
  "Classify every advertisement into a category and automatically attach the right compliance checklist for that category.",
  "Give every advertisement a unique, system-generated ID number — no manual numbering, no duplicates.",
  "Maintain a single register (the Advertisement Log) showing type, a review summary, a hyperlink to the fully filled-in checklist (“Page 2”), compliance comments and the final decision.",
  "Keep a backend Checklist Master that drives what gets checked per category, with a Copilot-assisted interface to help refresh it as regulations change — always with a human approval step before anything goes live.",
  "Give the CCO a dashboard with the key details needed to oversee the whole program at a glance.",
]));
children.push(p(
  "The guide that follows is written for a non-developer: every step names the exact screen, button and field to use. Section 3 gives you the architecture at a glance; Section 7 is the actual build, phase by phase. Read Section 3 and 4 first, then work through Section 7 top to bottom.",
  { after: 100 }
));
children.push(callout(
  "Bottom line:",
  "Everything described here is achievable by one motivated non-developer over roughly 1.5–2 focused work-weeks (part-time, spread around your normal job), using only licenses your organization most likely already has — see Section 4 for the one piece (Copilot) that may need a license check first."
));

// ---- 2. How the Tool Works ----
children.push(h1("How the Tool Works — The Review Journey", 2));
children.push(p("Here is the same tool described as a single walk-through, end to end:"));
children.push(...steps([
  "A reviewer opens the site and clicks New Ad Review. They select the advertisement's category and enter a short description.",
  "The moment that item is saved, the tool automatically assigns a unique Advertisement ID (e.g., ADV-2026-0001) and builds a checklist for that advertisement by pulling every active question for that category from the Checklist Master.",
  "The reviewer opens the Checklist Link — “Page 2” — and works through each question, marking it Compliant, Non-Compliant or Not Applicable, with comments and evidence where needed.",
  "The reviewer records an overall Review Summary, Compliance Comments and a Decision (Approved, Approved with Conditions, Rejected, or Escalated) back on the Advertisement Log.",
  "The submitter is notified automatically. The Advertisement Log — and the CCO's dashboard — update immediately, since they're reading the same data live.",
  "Periodically, the CCO uses the Copilot interface on the Checklist Admin page to review new or changed regulations and refresh the relevant checklist items, with every change going through an approval step before it becomes live.",
]));

// ---- 3. Architecture ----
children.push(pageBreak());
children.push(h1("Solution Architecture", 3));
children.push(p(
  "The diagram below shows the whole tool as four layers sitting inside one SharePoint site, plus the people who use it and the Copilot layer that keeps the checklist current. Nothing here requires custom code — every box is a standard Microsoft 365 building block."
));
children.push(...imageFigure(
  path.join(__dirname, "architecture.png"),
  624, 306,
  "Figure 1. Advertisement Compliance Review Tool — solution architecture."
));
children.push(h2("Layer by layer"));
children.push(...bullets([
  [{text:"Users. ",bold:true},{text:"You (the CCO) and your 2 team members, as SharePoint site members — no separate login, no extra app to install."}],
  [{text:"Presentation (SharePoint pages). ",bold:true},{text:"Four pages: a Home/Dashboard page for the CCO, a New Ad Review page, a Checklist Detail page (“Page 2”), and a Checklist Admin page."}],
  [{text:"Logic (Power Automate flows). ",bold:true},{text:"Five flows handle everything automatic: assigning IDs, building each checklist, sending notifications, and applying CCO-approved checklist updates."}],
  [{text:"Data (SharePoint lists & library). ",bold:true},{text:"Six data stores: Ad Categories, Checklist Master (+ Archive), Advertisement Log, Checklist Responses, and the Regulations Library."}],
  [{text:"Intelligence (Copilot). ",bold:true},{text:"A Copilot or Copilot Studio agent that reads the Regulations Library and the current Checklist Master, and drafts suggested checklist updates — always for the CCO to review and approve, never applied automatically."}],
]));

// ---- 4. Apps required ----
children.push(h1("Apps & Licenses You’ll Need", 4));
children.push(p("Everything in this design runs on Microsoft 365. The table below tells you what each piece is for and what it typically costs on top of a standard Microsoft 365 subscription — but licensing details vary by plan and do change, so confirm the Copilot row with your Microsoft 365 admin before you commit to it."));
children.push(dataTable(
  ["Area", "Microsoft 365 service", "Why you need it", "Typical licensing note"],
  [1500, 2000, 3380, 3200],
  appsRows
));
children.push(spacer(120));

// ---- 5. Data model ----
children.push(pageBreak());
children.push(h1("Data Model — The Lists You’ll Build", 5));
children.push(p("This is your backend checklist master and every other data store the tool needs, laid out column by column so you can build them exactly as described in Phase 2. Build them in the order shown, since later lists look up to earlier ones."));
for (const sec of dataModelSections) {
  children.push(h3(sec.name + "  (" + sec.kind + ")"));
  children.push(p(sec.purpose, { after: 100 }));
  children.push(dataTable(["Column name", "Type", "Purpose / notes"], [2200, 2200, 5680], sec.rows));
  children.push(spacer(200));
}

// ---- 6. Dashboard ----
children.push(pageBreak());
children.push(h1("The CCO Dashboard", 6));
children.push(p("The Home page is the CCO's dashboard — it needs no separate build effort beyond Phase 5.1, because it simply reads live from the Advertisement Log. Recommended contents:"));
children.push(...bullets([
  "A list view of recent advertisements grouped by Status (Draft / In Review / Completed), showing AdvertisementID, Category, Reviewed By, Decision and the Checklist Link.",
  "A count of ads by Decision (Approved / Rejected / Approved with Conditions / Pending / Escalated) using the Quick Chart web part.",
  "A count of ads by Category, to spot which advertisement types are generating the most review volume.",
  "A filtered view (or a second Quick Chart) highlighting anything In Review past your SLA — the same items Flow D reminds reviewers about.",
]));
children.push(p("Because this is a live list view rather than a static report, it is always current — there is nothing to refresh or re-run."));

// ---- 7. Build guide ----
children.push(pageBreak());
children.push(h1("Step-by-Step Build Guide", 7));
children.push(p("Work through the phases below in order. Each one names the exact screen, button and field to use — you do not need any coding background to follow it."));
for (const phase of phases) {
  children.push(h2(phase.heading));
  if (phase.intro) children.push(p(phase.intro, { after: 140 }));
  children.push(...renderBlocks(phase.blocks));
  children.push(spacer(120));
}

// ---- 8. Timeline ----
children.push(pageBreak());
children.push(h1("Effort & Timeline Estimate", 8));
children.push(p("Rough, solo estimates for someone comfortable with everyday Office apps but new to Power Automate — add a buffer for the learning curve on your first attempt."));
children.push(dataTable(["Phase", "What you're doing", "Estimated time", "Difficulty"], [1000, 3700, 2880, 2500], timelineRows));
children.push(spacer(140));
children.push(p("* Time for Phase 3 depends mainly on how much checklist content you already have typed up somewhere versus starting from paper/PDF source material.", { after: 100 }));
children.push(callout("Total for a working MVP:", "Phases 0–5, 8–9 and Tier 1 of Phase 7 — roughly 1.5–2 focused work-weeks, part-time. Phase 6 and Copilot Tiers 2–3 are optional accelerators you can add later without rebuilding anything."));

// ---- 9. Risks ----
children.push(pageBreak());
children.push(h1("Risks & Governance Considerations", 9));
children.push(...bullets([
  [{text:"Human-in-the-loop is essential. ",bold:true},{text:"Never let Copilot, or any automation, write directly to Checklist Master without the CCO's approval — Flow E's approval step is what actually enforces this, regardless of which Copilot tier you build."}],
  [{text:"Audit trail over convenience. ",bold:true},{text:"Checklist Responses snapshots the question wording, reference and version at review time (Section 5.5) precisely so a later checklist update can never silently rewrite what an old advertisement was actually reviewed against."}],
  [{text:"Never let a person type the Advertisement ID. ",bold:true},{text:"Generating it from the system (Flow A) is what guarantees it's always unique."}],
  [{text:"Access control. ",bold:true},{text:"Keep site membership to exactly the people who should see advertisement compliance decisions, and review membership periodically."}],
  [{text:"Confirm licensing before you build Phase 7 Tier 3. ",bold:true},{text:"Copilot Studio availability and cost vary by tenant and change over time — this guide cannot guarantee what's included in yours."}],
  [{text:"Scale. ",bold:true},{text:"SharePoint lists comfortably handle thousands of items; the indexed AdvertisementID column (Phase 2) keeps filtering fast well past that."}],
  [{text:"Continuity. ",bold:true},{text:"Export key lists to Excel periodically as an offline backup, and reassign site Ownership promptly if you or a teammate leaves — a site tied to a departed user's account can otherwise become hard to administer."}],
]));

// ---- 10. Appendices ----
children.push(pageBreak());
children.push(h1("Appendices", 10));

children.push(h2("Appendix A — Advertisement ID Convention"));
children.push(p("Recommended format: ADV-YYYY-####, e.g. ADV-2026-0001, generated automatically by Flow A from SharePoint's own item number (Phase 4). The sequence naturally restarts each calendar year because the year is baked into the prefix. Change “ADV” to match your own organization's naming convention if you prefer — just update the formula in Flow A to match."));

children.push(h2("Appendix B — Starter Categories & Sample Checklist Items"));
children.push(callout(
  "Illustrative only:",
  "The questions and references below are generic placeholders to help you get Checklist Master started structurally. They are not legal or regulatory advice. Replace the reference column with your organization's actual applicable requirements, confirmed with your legal/compliance function, before relying on this tool for real reviews.",
  AMBER
));
children.push(dataTable(["Category", "Sample checklist question", "Regulatory reference"], [2400, 4680, 3000], appendixBRows));
children.push(spacer(160));

children.push(h2("Appendix C — Glossary"));
children.push(dataTable(["Term", "Meaning"], [2400, 7680], glossaryRows));
children.push(spacer(160));

children.push(h2("Appendix D — Where to Go"));
children.push(dataTable(["Task", "Go to"], [4000, 6080], appendixDRows));

// =====================================================================
// STYLES + DOCUMENT
// =====================================================================

const doc = new Document({
  creator: "Compliance Function",
  title: "Advertisement Compliance Review Tool — Architecture & Build Guide",
  description: "Solution architecture and step-by-step non-developer build guide for an advertisement compliance review tool on Microsoft 365.",
  numbering: { config: numberingConfigs },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22, color: "1F2937" } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: NAVY, font: "Calibri" },
        paragraph: {
          spacing: { before: 420, after: 200 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 6 } },
        },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: BLUE, font: "Calibri" },
        paragraph: { spacing: { before: 320, after: 140 } },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: "374151", font: "Calibri" },
        paragraph: { spacing: { before: 220, after: 100 } },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              tabStops: [{ type: TabStopType.RIGHT, position: USABLE_WIDTH }],
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GRAY, space: 4 } },
              children: [
                new TextRun({ text: "Advertisement Compliance Review Tool — Confidential, Internal Use Only", size: 16, color: GRAY }),
                new TextRun({ text: "\t" }),
                new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GRAY }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

const outPath = path.join(__dirname, "Advertisement_Compliance_Review_Tool.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log("Wrote", outPath, buf.length, "bytes");
});
