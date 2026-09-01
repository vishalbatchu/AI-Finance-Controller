let allTransactions = [];
let sortKey = "date";
let sortDir = -1;
let searchTerm = "";

const CATEGORIES = ["Food", "Travel", "EMI", "Investment", "Shopping"];
const money = n => `₹${Number(n).toLocaleString("en-IN", {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
const pct = n => `${(Number(n) * 100).toFixed(1)}%`;

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (_) {
    const preview = text.replace(/\s+/g, " ").slice(0, 180);
    throw new Error(`Server returned HTTP ${response.status} instead of JSON: ${preview}`);
  }
  if (!response.ok) throw new Error(data.error || data.answer || `Request failed (${response.status})`);
  return data;
}

function showTransactions() {
  document.getElementById("transactionsPage").classList.remove("hidden");
  document.getElementById("dashboardPage").classList.add("hidden");
  window.scrollTo({top: 0, behavior: "smooth"});
}
function showDashboard() {
  document.getElementById("transactionsPage").classList.add("hidden");
  document.getElementById("dashboardPage").classList.remove("hidden");
  window.scrollTo({top: 0, behavior: "smooth"});
}

async function refresh() {
  const [summary, exceptions, transactions, activity, feedback] = await Promise.all([
    getJSON("/api/summary"), getJSON("/api/exceptions"), getJSON("/api/settlement_batch"),
    getJSON("/api/activity"), getJSON("/api/feedback")
  ]);
  allTransactions = transactions;

  document.getElementById("matchRate").textContent = pct(summary.match_rate);
  document.getElementById("totalTransactions").textContent = summary.total_transactions.toLocaleString("en-IN");
  document.getElementById("totalAmount").textContent = money(summary.total_amount);
  document.getElementById("settledCount").textContent = summary.status_counts.Settled || 0;
  document.getElementById("pendingCount").textContent = summary.status_counts.Pending || 0;
  document.getElementById("failedCount").textContent = summary.status_counts.Failed || 0;
  document.getElementById("exceptionCount").textContent = summary.exceptions_count;
  document.getElementById("autoResolved").textContent = summary.auto_resolved;
  document.getElementById("manualResolved").textContent = summary.manual_resolved;
  document.getElementById("unresolved").textContent = summary.unresolved;
  document.getElementById("feedbackCount").textContent = summary.feedback_count;

  const learning = summary.model_learning || {};
  document.getElementById("learningStatus").textContent = learning.learning_mode || "AUTOMATIC LEARNING: ON";
  document.getElementById("modelVersion").textContent = learning.version || "baseline";
  document.getElementById("humanExamplesLearned").textContent = learning.human_feedback_rows ?? summary.feedback_count ?? 0;
  document.getElementById("trainingRows").textContent = Number(learning.training_rows || 0).toLocaleString("en-IN");
  document.getElementById("lastLearned").textContent = learning.trained_at ? new Date(learning.trained_at).toLocaleString() : "Baseline";

  document.getElementById("frontBatchCount").textContent = summary.total_transactions.toLocaleString("en-IN");
  document.getElementById("tableLiveStatus").textContent = `SYNCED · ${new Date().toLocaleTimeString()}`;
  const matchPct = Math.max(0, Math.min(1, Number(summary.match_rate || 0)));
  const ring = document.getElementById("matchRing");
  if (ring) ring.style.background = `conic-gradient(var(--accent) ${matchPct * 360}deg, #242b32 0deg)`;
  const progress = document.getElementById("matchProgress");
  if (progress) progress.style.width = `${matchPct * 100}%`;
  const total = Number(summary.total_transactions || 0) || 1;
  const sc = summary.status_counts || {};
  [
    ["settledBar", sc.Settled || 0], ["pendingBar", sc.Pending || 0], ["failedBar", sc.Failed || 0]
  ].forEach(([id, value]) => { const el = document.getElementById(id); if (el) el.style.width = `${(Number(value) / total) * 100}%`; });

  renderExceptions(exceptions);
  renderTransactions();
  renderBreakdown(summary.category_counts);
  renderActivity(activity);
  renderPerformance(summary);
  renderFeedback(feedback);
}

function renderActivity(items) {
  const el = document.getElementById("activityLog");
  if (!items.length) {
    el.innerHTML = `<div class="muted-line">Agent activity will appear here as batches are processed.</div>`;
    return;
  }
  el.innerHTML = items.slice(0, 8).map(x => {
    const label = String(x.event || "").replaceAll("_", " ");
    return `<div class="activity-row"><span>${escapeHtml(new Date(x.timestamp).toLocaleTimeString())}</span><strong>${escapeHtml(label)}</strong><small>${activityDetail(x)}</small></div>`;
  }).join("");
}

function activityDetail(x) {
  if (x.transactions !== undefined) return `${x.transactions} transactions`;
  if (x.exceptions !== undefined) return `${x.exceptions} exceptions`;
  if (x.auto_resolved !== undefined) return `${x.auto_resolved} auto-resolved`;
  if (x.corrected_category) return `${x.original_category} → ${x.corrected_category}`;
  if (x.question) return String(x.question).slice(0, 50);
  return "recorded";
}

function renderPerformance(summary) {
  const m = summary.model_performance || {};
  document.getElementById("stressAccuracy").textContent = m.accuracy == null ? "—" : pct(m.accuracy);
  document.getElementById("performanceFoot").textContent = m.tested ? `${m.correct} / ${m.tested} unseen stress cases correct · ${m.exceptions} flagged for review` : "No stress-test report found.";
  const bars = document.getElementById("performanceBars");
  const entries = Object.entries(m.by_category || {});
  if (!entries.length) { bars.innerHTML = ""; return; }
  bars.innerHTML = entries.map(([cat, v]) => `<div class="perf-row"><span>${escapeHtml(cat)}</span><div class="bar-track"><div class="bar-fill" style="width:${(v.accuracy*100).toFixed(1)}%"></div></div><b>${pct(v.accuracy)}</b></div>`).join("");
}

function renderFeedback(items) {
  const el = document.getElementById("feedbackList");
  if (!items.length) {
    el.innerHTML = `<div class="muted-line">No human corrections yet. Correct an exception to create reusable training feedback.</div>`;
    return;
  }
  el.innerHTML = items.slice(0, 6).map(x => `<div class="feedback-row"><span>${escapeHtml(x.transaction_id)}</span><span>${escapeHtml(x.original_category)} → <strong>${escapeHtml(x.corrected_category)}</strong></span><span>${Number(x.model_confidence*100).toFixed(1)}% AI confidence</span></div>`).join("");
}

function renderExceptions(items) {
  const el = document.getElementById("exceptionsList");
  if (!items.length) {
    el.innerHTML = `<div class="exception-empty">✓ No low-confidence transactions in this batch.</div>`;
    return;
  }
  el.innerHTML = items.map(x => `
    <article class="exception-card">
      <div><div class="txn">${escapeHtml(x.transaction_id)}</div><div class="counterparty">${escapeHtml(x.counterparty)} · ${money(x.amount)}</div></div>
      <div><strong>${escapeHtml(x.category)}</strong><div class="counterparty">${escapeHtml(x.date)}</div></div>
      <div><span class="confidence-badge">${x.confidence_pct.toFixed(1)}%</span></div>
      <div><div class="reason">${escapeHtml(x.reason)}</div>${renderAlternatives(x.alternatives)}</div>
      <div class="correction-controls">
        <span class="correction-label">ADMIN OVERRIDE</span>
        <select class="exception-category" data-id="${escapeHtml(x.transaction_id)}">
          ${CATEGORIES.map(cat => `<option value="${cat}" ${cat === x.category ? "selected" : ""}>${cat}</option>`).join("")}
        </select>
        <button class="apply-category" data-id="${escapeHtml(x.transaction_id)}" type="button">RESOLVE</button>
      </div>
    </article>`).join("");
}

function renderAlternatives(items) {
  if (!items || items.length < 2) return "";
  return `<div class="alternatives">Next: ${escapeHtml(items[1].category)} ${pct(items[1].confidence)}</div>`;
}

async function applyCorrection(transactionId) {
  const select = document.querySelector(`.exception-category[data-id="${CSS.escape(transactionId)}"]`);
  const button = document.querySelector(`.apply-category[data-id="${CSS.escape(transactionId)}"]`);
  if (!select || !button) return;
  button.disabled = true;
  try {
    const data = await getJSON(`/api/transactions/${encodeURIComponent(transactionId)}/category`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({category: select.value})
    });
    const message = document.getElementById("exceptionUpdateMessage");
    message.textContent = data.learning_status === "LEARNED"
      ? `✓ ${data.message} The model immediately learned from this verified correction (${data.model_version}).`
      : `✓ ${data.message} The correction was saved. Model learning will retry later. ${data.learning_error ? `Reason: ${data.learning_error}` : ""}`;
    message.className = "exception-update-message success";
    await refresh();
  } catch (err) {
    document.getElementById("exceptionUpdateMessage").textContent = err.message;
  } finally { button.disabled = false; }
}

document.getElementById("exceptionsList").addEventListener("click", e => {
  const button = e.target.closest(".apply-category");
  if (button) applyCorrection(button.dataset.id);
});

function filteredTransactions() {
  const cat = document.getElementById("categoryFilter").value;
  const status = document.getElementById("statusFilter").value;
  const q = searchTerm.toLowerCase();
  return allTransactions.filter(x => !cat || x.category === cat).filter(x => !status || x.status === status).filter(x => !q || [x.transaction_id, x.counterparty, x.category, x.status, x.date].some(v => String(v ?? "").toLowerCase().includes(q))).sort((a,b) => {
    let av=a[sortKey], bv=b[sortKey];
    if (sortKey === "amount" || sortKey === "category_confidence") { av=Number(av); bv=Number(bv); }
    else { av=String(av); bv=String(bv); }
    return av < bv ? -1*sortDir : av > bv ? 1*sortDir : 0;
  });
}

function renderTransactions() {
  const rows = filteredTransactions();
  document.getElementById("transactionsBody").innerHTML = rows.map(x => `<tr>
    <td><strong>${escapeHtml(x.transaction_id)}</strong></td><td>${escapeHtml(x.date)}</td><td>${escapeHtml(x.counterparty)}</td>
    <td class="amount">${money(x.amount)}</td><td>${escapeHtml(x.category)}</td>
    <td class="conf ${Number(x.category_confidence)<.6 ? "low-conf" : ""}">${pct(x.category_confidence)}</td>
    <td><span class="status status-${x.status.toLowerCase()}">${escapeHtml(x.status.toUpperCase())}</span></td>
  </tr>`).join("");
  document.getElementById("tableFooter").textContent = `Showing ${rows.length} of ${allTransactions.length} transactions`;
}

function renderBreakdown(counts) {
  const entries = Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  const max = Math.max(...entries.map(x=>x[1]),1);
  document.getElementById("breakdown").innerHTML = entries.map(([label,count]) => `<div class="bar-row"><span class="bar-label">${escapeHtml(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${(count/max)*100}%"></div></div><span class="bar-count">${count}</span></div>`).join("");
}

function escapeHtml(value) { return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c])); }

async function askQuestion(question) {
  const input=document.getElementById("question"), answer=document.getElementById("qaAnswer");
  input.value=question; answer.classList.remove("hidden"); answer.textContent="Computing from the current batch…";
  try { const data=await getJSON("/api/qa",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question})}); answer.textContent=data.answer; }
  catch(err){ answer.textContent=err.message; }
}

document.getElementById("dashboardBtn").addEventListener("click",showDashboard);
document.getElementById("backToTransactions").addEventListener("click",showTransactions);
document.getElementById("categoryFilter").addEventListener("change",renderTransactions);
document.getElementById("transactionSearch").addEventListener("input", e => { searchTerm = e.target.value.trim(); renderTransactions(); });
document.getElementById("clearFilters").addEventListener("click", () => { document.getElementById("categoryFilter").value = ""; document.getElementById("statusFilter").value = ""; document.getElementById("transactionSearch").value = ""; searchTerm = ""; renderTransactions(); });
document.addEventListener("keydown", e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); document.getElementById("transactionSearch").focus(); } });
document.getElementById("statusFilter").addEventListener("change",renderTransactions);
document.querySelectorAll("th[data-key]").forEach(th=>th.addEventListener("click",()=>{const key=th.dataset.key;if(sortKey===key)sortDir*=-1;else{sortKey=key;sortDir=1;}renderTransactions();}));

document.getElementById("qaForm").addEventListener("submit",e=>{e.preventDefault();const q=document.getElementById("question").value.trim();if(q)askQuestion(q);});
document.getElementById("suggestedQuestions").addEventListener("click",e=>{const b=e.target.closest("button[data-q]");if(b)askQuestion(b.dataset.q);});

document.getElementById("csvFile").addEventListener("change",e=>{document.getElementById("fileName").textContent=e.target.files[0]?.name||"Choose CSV";});
document.getElementById("uploadForm").addEventListener("submit",async e=>{e.preventDefault();const file=document.getElementById("csvFile").files[0],message=document.getElementById("uploadMessage");if(!file){message.textContent="Choose a CSV first.";return;}const form=new FormData();form.append("file",file);message.textContent="Classifying and evaluating batch…";try{const data=await getJSON("/api/upload",{method:"POST",body:form});message.textContent=`Loaded and classified ${data.count} transaction(s).`;await refresh();showTransactions();}catch(err){message.textContent=err.message;}});

document.getElementById("exportFeedback").addEventListener("click",()=>{window.location.href="/api/feedback/export";});
document.getElementById("retrainBtn").addEventListener("click",async()=>{const msg=document.getElementById("retrainMessage");msg.textContent="Retraining with verified human corrections…";try{const data=await getJSON("/api/retrain",{method:"POST"});msg.textContent=`✓ ${data.message} ${data.feedback_rows} verified human corrections incorporated. Model: ${data.model_version || "updated"}.`;await refresh();}catch(err){msg.textContent=err.message;}});

refresh().catch(err=>{document.getElementById("uploadMessage").textContent=`Dashboard error: ${err.message}`;});
