const $ = (id) => document.getElementById(id);
let current = null,
  page = 1,
  listing = false,
  activeTab = "ask",
  documents = [],
  listSignature = "";
const names = [
  "vendor_name",
  "invoice_number",
  "invoice_date",
  "due_date",
  "currency",
  "subtotal",
  "tax",
  "grand_total",
];
let chatAvailable = false;
let availabilityMessage = null;
let asking = false;
const labels = {
  vendor_name: "Vendor",
  invoice_number: "Invoice number",
  invoice_date: "Invoice date",
  due_date: "Due date",
  currency: "Currency",
  subtotal: "Subtotal",
  tax: "Tax",
  grand_total: "Grand total",
  description: "Description",
  quantity: "Quantity",
  unit_price: "Unit price",
  amount: "Amount",
};
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let error;
    try {
      error = await response.json();
    } catch {
      error = { detail: response.statusText };
    }
    throw Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
function text(tag, value, parent) {
  const el = document.createElement(tag);
  el.textContent = value;
  parent?.append(el);
  return el;
}
function warning(value, parent) {
  text("p", value, parent).className = "warning";
}
function switchTab(name) {
  activeTab = name;
  for (const tab of ["ask", "invoice", "document"]) {
    $("tab-" + tab).setAttribute("aria-selected", String(tab === name));
    $("tab-" + tab).tabIndex = tab === name ? 0 : -1;
    $("panel-" + tab).hidden = tab !== name;
  }
  updateBadge();
}
function updateBadge() {
  if (!current) return;
  const ready = current.status === "ready";
  $("statusBadge").textContent = ready
    ? activeTab === "invoice"
      ? current.review_status === "approved"
        ? "Approved"
        : "Needs review"
      : "Ready"
    : current.status === "failed"
      ? "Could not process"
      : "Processing";
  $("statusBadge").classList.toggle(
    "attention",
    current.status === "failed" ||
      (ready &&
        activeTab === "invoice" &&
        current.review_status !== "approved"),
  );
}
function renderDocuments() {
  $("documents").replaceChildren();
  if (!documents.length) {
    text(
      "p",
      "Your uploaded files will appear here.",
      $("documents"),
    ).className = "hint";
    return;
  }
  for (const doc of documents) {
    const button = document.createElement("button");
    button.className =
      "document" +
      (current?.document_id === doc.document_id ? " selected" : "");
    button.setAttribute(
      "aria-pressed",
      String(current?.document_id === doc.document_id),
    );
    button.title = doc.filename;
    text("span", doc.filename, button).className = "document-name";
    text(
      "span",
      doc.status === "ready"
        ? doc.review_status === "approved"
          ? "Approved"
          : "Ready to use"
        : doc.status === "failed"
          ? "Upload needs attention"
          : "Processing…",
      button,
    ).className = "document-state";
    button.onclick = () => {
      page = 1;
      load(doc.document_id);
    };
    $("documents").append(button);
  }
}
async function checkChatStatus() {
  try {
    const status = await api("/api/chat/status");
    const changed =
      chatAvailable !== status.available ||
      availabilityMessage !== status.message;
    chatAvailable = status.available;
    availabilityMessage = status.message;
    if (!asking && (changed || !chatAvailable))
      $("chatStatus").textContent = status.message;
  } catch {
    chatAvailable = false;
    availabilityMessage = null;
    if (!asking)
      $("chatStatus").textContent =
        "Chat is unavailable right now. Please try again shortly.";
  }
  $("askButton").disabled =
    !chatAvailable || current?.status !== "ready" || asking;
}
async function list() {
  if (listing) return;
  listing = true;
  try {
    const data = await api("/api/documents");
    documents = data.documents;
    const signature = JSON.stringify(documents);
    if (signature !== listSignature) {
      listSignature = signature;
      renderDocuments();
    }
    if (current && !["ready", "failed"].includes(current.status))
      await load(current.document_id);
  } catch (error) {
    $("uploadStatus").textContent = error.message;
  } finally {
    listing = false;
  }
}
function renderOriginal() {
  $("machine").replaceChildren();
  for (const name of names) {
    const row = document.createElement("div");
    row.className = "original-row";
    text("span", labels[name], row);
    text("span", current.original?.fields[name]?.raw ?? "Not found", row);
    $("machine").append(row);
  }
  for (const [index, item] of (current.original?.line_items || []).entries()) {
    const row = document.createElement("div");
    row.className = "original-row";
    text("span", `Item ${index + 1}`, row);
    text(
      "span",
      ["description", "quantity", "unit_price", "amount"]
        .map((k) => item[k].raw ?? "—")
        .join(" · "),
      row,
    );
    $("machine").append(row);
  }
  $("audit").replaceChildren();
  for (const event of current.audit || []) {
    const row = document.createElement("p");
    text(
      "strong",
      event.action === "approve" ? "Invoice approved" : "Changes saved",
      row,
    );
    text("span", " · " + new Date(event.timestamp).toLocaleString(), row);
    if (event.acknowledgement) text("p", event.acknowledgement, row);
    $("audit").append(row);
  }
  if (!current.audit?.length) text("p", "No changes yet.", $("audit"));
}
async function load(id) {
  try {
    const record = await api("/api/documents/" + id);
    const changed = current?.document_id !== id;
    current = record;
    page = Math.min(page, current.pages.length || 1);
    $("welcome").hidden = true;
    $("documentWorkspace").hidden = false;
    $("title").textContent = current.filename;
    const messages = {
      queued: "Your document is waiting to process.",
      extracting: "Reading your document…",
      indexing: "Preparing your document for questions…",
    };
    $("status").textContent =
      current.status === "ready"
        ? ""
        : messages[current.status] || current.message;
    $("original").href = `/api/documents/${id}/original`;
    $("json").href = `/api/documents/${id}/export?format=json`;
    $("csv").href = `/api/documents/${id}/export?format=csv`;
    $("fields").replaceChildren();
    $("items").replaceChildren();
    $("warnings").replaceChildren();
    const warnings = [
      ...(current.validation_warnings || []),
      ...(current.duplicate_warnings || []),
    ];
    warnings.forEach((w) => warning(w, $("warnings")));
    $("warningDetails").hidden = !warnings.length;
    $("warningSummary").textContent =
      `${warnings.length} ${warnings.length === 1 ? "item needs" : "items need"} your attention`;
    const ready = current.status === "ready" && !!current.reviewed;
    if (ready) {
      names.forEach((name) =>
        field(name, current.reviewed.fields[name], $("fields")),
      );
      current.reviewed.line_items.forEach(addItem);
      renderOriginal();
    }
    $("review").hidden = !ready;
    $("reviewPending").hidden = ready;
    $("reviewDetails").hidden = !ready;
    $("exports").hidden = !ready;
    $("askButton").disabled =
      !chatAvailable || current.status !== "ready" || asking;
    $("reviewPending").textContent =
      current.status === "failed"
        ? "Upload a valid document to review its invoice details."
        : "Invoice details will appear when processing finishes.";
    if (changed) {
      $("note").value = "";
      $("answerArea").hidden = true;
      $("chatHint").hidden = false;
      $("answer").textContent = "";
      $("citations").replaceChildren();
      switchTab("ask");
    }
    updateBadge();
    renderDocuments();
    showPage();
    return true;
  } catch (error) {
    $("uploadStatus").textContent = error.message;
    return false;
  }
}
function showPage(evidence = "") {
  const p = current?.pages[page - 1];
  $("pageLabel").textContent = p
    ? `Page ${page} of ${current.pages.length}`
    : "No pages yet";
  $("previous").disabled = !p || page <= 1;
  $("next").disabled = !p || page >= current.pages.length;
  $("preview").hidden = !p || current.extension === ".txt";
  $("previewPending").hidden = !!p;
  $("textDetails").hidden = !p;
  $("textDetails").open = current?.extension === ".txt";
  $("sourceQuote").hidden = !evidence;
  $("sourceQuote").textContent = evidence;
  $("highlights").replaceChildren();
  $("pageText").textContent = p?.text || "";
  $("pageWarnings").replaceChildren();
  for (const w of p?.warnings || []) warning(w, $("pageWarnings"));
  if (!p) return;
  $("preview").src = `/api/documents/${current.document_id}/pages/${page}`;
  for (const block of p.blocks || []) {
    if (
      !evidence ||
      p.method !== "native" ||
      !evidence.includes(block.text) ||
      p.blocks.filter((other) => other.text === block.text).length !== 1 ||
      !block.bbox
    )
      continue;
    const el = document.createElement("div");
    el.className = "box";
    const [x, y, x2, y2] = block.bbox;
    Object.assign(el.style, {
      left: x * 100 + "%",
      top: y * 100 + "%",
      width: (x2 - x) * 100 + "%",
      height: (y2 - y) * 100 + "%",
    });
    $("highlights").append(el);
  }
}
function field(name, value, parent, showEvidence = true) {
  const box = document.createElement("div");
  box.className = "field";
  box.dataset.name = name;
  box._value = value;
  const label = text("label", labels[name], box);
  const input = document.createElement("input");
  input.value = value.normalized ?? value.raw ?? "";
  input.maxLength = 2000;
  input.dataset.original = input.value;
  input.placeholder = ["invoice_date", "due_date"].includes(name)
    ? "YYYY-MM-DD"
    : name === "currency"
      ? "e.g. USD"
      : "Not found";
  label.append(input);
  const footer = document.createElement("div");
  footer.className = "field-footer";
  if (value.status !== "present" && name !== "due_date")
    text(
      "span",
      value.status === "ambiguous" ? "Please check" : "Missing",
      footer,
    ).className = "field-status";
  // Keep supporting text one click away without repeating it under every input.
  for (const evidence of showEvidence ? value.evidence || [] : []) {
    const b = text(
      "button",
      `View source · p. ${evidence.page_number}`,
      footer,
    );
    b.type = "button";
    b.className = "evidence";
    b.title = evidence.text;
    b.onclick = () => {
      page = evidence.page_number;
      showPage(evidence.text);
      switchTab("document");
    };
  }
  if (footer.children.length) box.append(footer);
  parent.append(box);
}
function addItem(item = {}) {
  const box = document.createElement("div");
  box.className = "item";
  for (const name of ["description", "quantity", "unit_price", "amount"])
    field(
      name,
      item[name] || {
        raw: null,
        normalized: null,
        status: "missing",
        evidence: [],
      },
      box,
      name === "description",
    );
  const remove = text("button", "×", box);
  remove.type = "button";
  remove.className = "remove-item";
  remove.setAttribute("aria-label", "Remove line item");
  remove.onclick = () => box.remove();
  $("items").append(box);
}
function readFields(parent) {
  return Object.fromEntries(
    [...parent.children]
      .filter((el) => el.className === "field")
      .map((el) => {
        const input = el.querySelector("input");
        const old = el._value;
        return [
          el.dataset.name,
          input.value === input.dataset.original
            ? old
            : {
                ...old,
                normalized: input.value.trim() || null,
                status: input.value.trim() ? "present" : "missing",
              },
        ];
      }),
  );
}
$("chooseFile").onclick = $("welcomeUpload").onclick = () => $("file").click();
$("file").onchange = () => {
  if ($("file").files.length) $("upload").requestSubmit();
};
$("upload").onsubmit = async (event) => {
  event.preventDefault();
  $("uploadStatus").textContent = "Uploading…";
  $("chooseFile").disabled = true;
  try {
    const form = new FormData();
    form.append("file", $("file").files[0]);
    const result = await api("/api/ingest", { method: "POST", body: form });
    $("uploadStatus").textContent = "";
    page = 1;
    await load(result.document_id);
    await list();
  } catch (error) {
    $("uploadStatus").textContent = error.message;
  } finally {
    $("chooseFile").disabled = false;
    $("file").value = "";
  }
};
for (const [index, name] of ["ask", "invoice", "document"].entries()) {
  const button = $("tab-" + name);
  button.onclick = () => switchTab(name);
  button.onkeydown = (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % 3;
    else if (event.key === "ArrowLeft") next = (index + 2) % 3;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = 2;
    else return;
    event.preventDefault();
    const target = ["ask", "invoice", "document"][next];
    switchTab(target);
    $("tab-" + target).focus();
  };
}
$("previous").onclick = () => {
  if (page > 1) {
    page--;
    showPage();
  }
};
$("next").onclick = () => {
  if (page < current?.pages.length) {
    page++;
    showPage();
  }
};
$("addItem").onclick = () => addItem();
$("review").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const invoice = {
      fields: readFields($("fields")),
      line_items: [...$("items").children].map(readFields),
      unsupported_adjustments: current.reviewed.unsupported_adjustments,
    };
    await api(`/api/documents/${current.document_id}/review`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        invoice,
        version: current.version,
        action: event.submitter?.value || "save",
        acknowledgement: $("note").value,
      }),
    });
    await load(current.document_id);
    $("status").textContent =
      event.submitter?.value === "approve"
        ? "Invoice approved."
        : "Changes saved.";
    await list();
  } catch (error) {
    $("status").textContent = error.message;
    if (error.message.includes("warnings")) {
      $("warningDetails").open = true;
      $("note").closest("details").open = true;
      $("note").focus();
    }
  }
};
$("chat").onsubmit = async (event) => {
  event.preventDefault();
  if (!chatAvailable) {
    await checkChatStatus();
    if (!chatAvailable) return;
  }
  asking = true;
  $("answerArea").hidden = true;
  $("chatHint").hidden = true;
  $("answer").textContent = "";
  $("chatStatus").textContent = "Looking through your documents…";
  $("citations").replaceChildren();
  $("askButton").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: "browser",
        message: $("question").value,
        document_ids: $("allDocs").checked
          ? null
          : current
            ? [current.document_id]
            : [],
      }),
    });
    $("answerArea").hidden = false;
    $("chatStatus").textContent = "";
    $("answer").textContent = data.response;
    for (const source of data.sources) {
      const b = text(
        "button",
        `[${source.citation}] ${source.document} · p. ${source.page_number}`,
        $("citations"),
      );
      b.className = "evidence";
      b.onclick = async () => {
        if (
          current?.document_id !== source.document_id &&
          !(await load(source.document_id))
        )
          return;
        page = source.page_number;
        showPage(source.text_preview);
        switchTab("document");
      };
    }
  } catch (error) {
    $("answerArea").hidden = true;
    $("answer").textContent = "";
    $("citations").replaceChildren();
    $("chatStatus").textContent = error.message;
  } finally {
    asking = false;
    $("askButton").disabled = !chatAvailable || current?.status !== "ready";
  }
};
$("summary").onclick = async () => {
  try {
    const data = await api("/api/invoices/summary");
    $("totals").replaceChildren();
    for (const [currency, value] of Object.entries(data.currencies))
      text(
        "p",
        `${currency} ${value.total} · ${value.documents.length} approved ${value.documents.length === 1 ? "invoice" : "invoices"}`,
        $("totals"),
      );
    if (!Object.keys(data.currencies).length)
      text("p", "Approve an invoice to see its total here.", $("totals"));
    else
      text(
        "p",
        "Includes all approved invoices, including any duplicates.",
        $("totals"),
      );
  } catch (error) {
    $("totals").textContent = error.message;
  }
};
switchTab("ask");
list();
checkChatStatus();
setInterval(list, 2500);
setInterval(checkChatStatus, 10000);
