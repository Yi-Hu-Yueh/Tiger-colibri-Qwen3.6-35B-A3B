const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const kv = items => `<div class="kv">${items.map(([k,v]) => `<span>${esc(k)}</span><strong>${esc(v)}</strong>`).join("")}</div>`;
const gib = bytes => bytes == null ? "—" : `${(bytes / 1073741824).toFixed(2)} GiB`;

function badge(id, status) {
  const element = $(`#${id} .badge`); if (!element) return;
  const label = (status || "NOT RUN").toUpperCase();
  element.textContent = label;
  element.className = `badge ${label.toLowerCase().replaceAll(" ", "-")}`;
}
function toast(message) { const el=$("#toast"); el.textContent=message; el.classList.add("show"); setTimeout(()=>el.classList.remove("show"),3000); }
async function api(path, method="GET", body) {
  let response;
  try {
    response = await fetch(path,{method,headers:body?{"Content-Type":"application/json"}:{},body:body?JSON.stringify(body):undefined});
  } catch (error) {
    throw new Error(`Cannot reach the local FastAPI server (${error.message || "network failure"}).`);
  }
  const text = await response.text();
  let data = {};
  if (text) {
    try { data = JSON.parse(text); }
    catch (_error) { data = {error: text.slice(0, 500)}; }
  }
  if (!response.ok) {
    const message = data.error || data.detail || `HTTP ${response.status}`;
    const code = data.code ? `${data.code}: ` : "";
    throw new Error(`${code}${message}`);
  }
  return data;
}
async function busy(button, work) { button.disabled=true; const old=button.textContent; button.textContent="Working…"; try{return await work();}catch(e){toast(e.message);throw e;}finally{button.disabled=false;button.textContent=old;} }

async function waitForModelReady(timeoutMs=190000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const current = await api("/api/chat/status");
    setChatStatus(current.status, current.error);
    if (current.status === "READY") return current;
    if (current.status === "ERROR") {
      const code = current.error_code ? `${current.error_code}: ` : "";
      throw new Error(`${code}${current.error || "Model startup failed."}`);
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error("MODEL_START_TIMEOUT: Qwen3.6 did not reach READY within 190 seconds.");
}

async function hardware() {
  const d=await api("/api/hardware"); badge("hardware","PASS");
  $("#hardware .content").innerHTML=kv([["CPU",d.cpu.model],["Physical cores",d.cpu.physical_cores],["Logical processors",d.cpu.logical_processors],["Architecture",d.cpu.architecture],["Windows",d.windows.edition],["Version / build",`${d.windows.version} / ${d.windows.build}`]]);
  const ram=d.ram; badge("ram",ram.total_gib?"PASS":"WARN"); $("#ram .content").innerHTML=kv([["Total",`${ram.total_gib} GiB`],["Available",`${ram.available_gib} GiB`],["Used",`${ram.used_gib} GiB (${ram.percent_used}%)`]]);
  const devices=d.gpu.devices||[]; badge("gpu","WARN"); $("#gpu .content").innerHTML=devices.length?devices.map(g=>kv([["GPU",g.name],["VRAM",g.vram_mib?`${g.vram_mib} MiB`:"Unknown"],["Driver",g.driver_version],["nvidia-smi",d.gpu.nvidia_smi_available?"Available":"Not available"],["Driver-reported CUDA",d.gpu.cuda_runtime_reported||"Not reported"],["CUDA Toolkit","Not validated; deferred to Phase 4"]])).join("<hr>"):kv([["GPU","No device detected"]]);
  badge("python",d.python.version?.startsWith("3.11.")?"PASS":"FAIL"); $("#python .content").innerHTML=kv([["Web-server version",d.python.version],["Web-server executable",d.python.executable],["Required","Python 3.11.x"]]); return d;
}

function renderStorageConfiguration(d) {
  const p=d.project_storage||{}, dl=d.download_root_validation||{}, rt=d.model_runtime_validation;
  badge("storage-config",dl.status==="PASS"?"PASS":"FAIL");
  const projectDisk=(d.physical_disks||[]).find(x=>x.disk_number===p.disk_number);
  $("#storage-config .content").innerHTML=`<div class="storage-roots">
    <article><h3>PROJECT ROOT</h3>${kv([["Path",d.project_root],["Physical disk",projectDisk?.friendly_name],["Media class",projectDisk?.classification?.classification],["Free space",gib(p.free_bytes)],["Purpose","source code / application"],["Status","PASS"]])}</article>
    <article><h3>DOWNLOAD ROOT</h3>${kv([["Path",d.download_root],["Physical device",dl.physical_device],["Media type",dl.media_type],["Free space",gib(dl.free_bytes)],["Purpose","model acquisition / archive"],["Status",d.download_root_status],["Warning",dl.warning]])}</article>
    <article><h3>MODEL RUNTIME ROOT</h3>${kv([["Path",d.model_runtime_root||"UNSET"],["Physical device",rt?.physical_device||"—"],["Media class",d.runtime_storage_class||"—"],["Free space",rt?gib(rt.free_bytes):"—"],["Purpose","actual Colibrì inference / expert streaming"],["Status",d.runtime_root_status]])}</article>
  </div>`;
}

async function storage() {
  const d=await api("/api/storage"); badge("storage",d.physical_disks?.length?"PASS":"FAIL");
  const disks=d.physical_disks||[], volumes=d.volumes||[];
  $("#storage .content").innerHTML=`<table><thead><tr><th>Disk</th><th>Device / model</th><th>Media / bus</th><th>Class</th><th>Evidence</th><th>Size</th></tr></thead><tbody>${disks.map(x=>`<tr><td>${esc(x.disk_number)}</td><td>${esc(x.friendly_name)}<br><span class="muted">${esc(x.model)}</span></td><td>${esc(x.media_type)} / ${esc(x.bus_type)}<br><span class="muted">${esc(x.interface_type)}</span></td><td>${esc(x.classification?.classification)}</td><td>${esc((x.classification?.evidence||[]).join("; "))}</td><td>${gib(x.size_bytes)}</td></tr>`).join("")}</tbody></table><h3>Mapped local volumes</h3><table><tbody>${volumes.map(v=>`<tr><td>${esc(v.drive_letter)}</td><td>Disk ${esc(v.disk_number)} / partition ${esc(v.partition_number)}</td><td>${esc(v.filesystem)}</td><td>${gib(v.free_bytes)} free</td><td>${esc(v.drive_type)}</td></tr>`).join("")}</tbody></table>`;
  renderStorageConfiguration(d); return d;
}

async function ollama(){const d=await api("/api/ollama");badge("ollama",d.detected?"PASS":"FAIL");$("#ollama .content").innerHTML=kv([["Detected",d.detected?"Yes":"No"],["Executable",d.executable_path],["Process detected",d.process_detected?"Yes":"No"],["API reachable",d.api_reachable?"Yes":"No"],["Version",d.api_version||d.cli_version],["Models",(d.installed_models||[]).map(m=>m.name).join(", ")],["Protection",d.protection]]);return d;}
async function release(){const d=await api("/api/colibri/release");badge("release",d.status);$("#release .content").innerHTML=kv([["Release",d.tag],["Published",d.published_at],["Windows asset",d.windows_asset?.name],["SHA256SUMS",d.checksum_asset?.name||"Not published"],["URL",d.release_url]]);return d;}

function renderSpaceAnalysis(d){badge("space-analysis",d.status==="PASS"?"PASS":d.status);if(d.status==="NOT RUN"){$("#space-analysis .content").innerHTML='<p class="muted">No analysis has been run.</p>';return;}const drive=d.drive||{},dirs=(d.largest_directories||[]).slice(0,30),files=(d.largest_files||[]).slice(0,100);$("#space-analysis .content").innerHTML=kv([["C: total",gib(drive.total_bytes)],["Used",gib(drive.used_bytes)],["Free",gib(drive.free_bytes)],["Additional for 30 GiB",`${drive.additional_for_30_gib} GiB`],["Additional for 45–50 GiB",`${drive.additional_for_45_gib}–${drive.additional_for_50_gib} GiB`]])+`<h3>Largest measured directories</h3><table><tbody>${dirs.map(x=>`<tr><td>${esc(x.path)}</td><td>${gib(x.size_bytes)}</td><td>${esc(x.risk_label)}</td></tr>`).join("")}</tbody></table><h3>Large files ≥500 MiB</h3><table><tbody>${files.map(x=>`<tr><td>${esc(x.path)}</td><td>${gib(x.size_bytes)}</td><td>${esc(x.risk_label)}</td></tr>`).join("")}</tbody></table>`;}

async function gate(){const d=await api("/api/phase/1");badge("gate",d.verdict);const overall=$("#overall-badge");overall.textContent=d.phase1_status||d.verdict;overall.className=`badge ${d.verdict.toLowerCase().replaceAll(" ","-")}`;$("#gate .content").innerHTML=kv([["Phase 1 status",d.phase1_status||d.verdict],["Acquisition readiness",d.acquisition_readiness],["Inference readiness",d.inference_readiness]])+`<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>${d.checks.map(c=>`<tr><td>${esc(c.name)}</td><td><span class="badge ${c.status.toLowerCase().replaceAll(" ","-")}">${esc(c.status)}</span></td><td>${esc(c.message)}</td></tr>`).join("")}</tbody></table>`;return d;}

function renderModel(d){badge("model-acquisition",d.download_status);const v=d.verification||{},s=d.structure_validation||{};$("#model-acquisition .content").innerHTML=kv([["Repository",d.repo_id],["Pinned revision",d.revision],["Remote expected bytes",d.remote_expected_bytes==null?"Unavailable":`${d.remote_expected_bytes.toLocaleString()} (${gib(d.remote_expected_bytes)})`],["Local bytes",`${d.local_bytes.toLocaleString()} (${gib(d.local_bytes)})`],["Remote / local files",`${d.remote_file_count??"—"} / ${d.local_file_count}`],["Download status",d.download_status],["Integrity status",d.integrity_status],["D: free before",gib(d.free_space_before_bytes)],["D: free after",gib(d.free_space_after_bytes)],["Started",d.download_started_at],["Completed",d.download_completed_at],["Verified",d.verification_timestamp],["File list",v.file_list_match],["File sizes",v.file_sizes_match],["Hashes",`${v.hash_files_verified??0}/${v.hash_files_available??0} files`],["expert_gs",s.expert_gs],["Model family",s.geometry?.model_type||s.geometry?.text_model_type],["Global checksum manifest",d.global_checksum_manifest||"Not published"],["Warnings",(d.warnings||[]).join(" | ")],["Errors",(d.verification_errors||[]).join(" | ")]]);return d;}
async function modelStatus(){return renderModel(await api("/api/model/qwen36/status"));}
function renderOllamaStorage(d){badge("ollama-storage",d.status);const b=d.baseline||{},a=d.after||{},dst=d.destination_inventory||{};$("#ollama-storage .content").innerHTML=kv([["Ollama version",d.ollama_version],["OLLAMA_MODELS",d.persisted_ollama_models],["Model storage",d.destination],["Storage device",d.storage_device||"ST2000LM007-1R8174"],["Media type",d.media_type||"HDD"],["Model count",d.post_model_count??b.model_count],["Model footprint",gib(dst.total_bytes??b.source_bytes)],["Migration",d.status],["C: free before / after",`${gib(b.c_free_bytes)} / ${gib(a.c_free_bytes)}`],["D: free before / after",`${gib(b.d_free_bytes)} / ${gib(a.d_free_bytes)}`],["Warning",d.warning||"D: is HDD-backed; cold model loads may be slower."]]);return d;}
async function ollamaStorageStatus(){return renderOllamaStorage(await api("/api/ollama/storage"));}

document.addEventListener("click",async e=>{const b=e.target.closest("button[data-action]");if(!b)return;await busy(b,async()=>{switch(b.dataset.action){
  case "scan-all":await Promise.all([hardware(),storage(),ollama()]);toast("Read-only scan complete");break;
  case "ollama":await ollama();break; case "release":await release();break;
  case "analyze-space":renderSpaceAnalysis(await api("/api/storage/analyze-space","POST"));toast("Read-only C: analysis complete");break;
  case "benchmark":{const d=await api("/api/storage/benchmark","POST",{size_mib:Number($("#benchmark-size").value)});badge("benchmark",d.status);$("#benchmark .content").innerHTML=kv([["Target",d.target],["Write",`${d.write_mbps} MB/s`],["Read",`${d.read_mbps} MB/s`],["Caveat",d.note]]);break;}
  case "download":{const d=await api("/api/colibri/download","POST");badge("install",d.status);$("#install .content").innerHTML=kv([["ZIP",d.zip_path],["SHA-256",d.checksum?.status],["Message",d.checksum?.message||d.error]]);break;}
  case "extract":{const d=await api("/api/colibri/extract","POST");badge("install",d.status);$("#install .content").innerHTML=kv([["Runtime",d.runtime_path],["qwen36.exe",d.qwen36_path]]);break;}
  case "smoke":{const d=await api("/api/colibri/smoke","POST");badge("smoke",d.status);$("#smoke .content").innerHTML=kv([["Classification",d.classification],["Result",d.status],["Note",d.note||d.error]]);break;}
  case "gate":await gate();break;
  case "model-download":renderModel(await api("/api/model/qwen36/download","POST"));break;
  case "model-verify":renderModel(await api("/api/model/qwen36/verify","POST"));break;
}}).catch(()=>{});});

api("/api/storage/space-analysis").then(renderSpaceAnalysis).catch(()=>{});
storage().catch(()=>{});
gate().catch(()=>{});
modelStatus().catch(()=>{});
ollamaStorageStatus().catch(()=>{});

const chatHistory = [];
let chatState = "STOPPED";
let activeChatController = null;
let activeGenerationId = null;
let cancellationInFlight = false;

function updateSendAvailability() {
  const canSend = chatState === "READY" && !activeChatController && Boolean($("#chat-input").value.trim());
  $("#chat-send").disabled = !canSend;
}

function setChatStatus(status, error="") {
  chatState = status;
  const badgeEl = $("#chat-status");
  badgeEl.textContent = status;
  badgeEl.className = `badge ${status.toLowerCase()}`;
  const overall = $("#overall-badge");
  overall.textContent = `QWEN ${status}`;
  overall.className = `badge ${status.toLowerCase()}`;
  $("#chat-error").textContent = error || "";
  $("#chat-start").disabled = ["STARTING","READY","SEARCHING","GENERATING"].includes(status);
  $("#chat-stop").disabled = status === "STOPPED";
  const canCancelSearch = status === "SEARCHING" && Boolean(activeChatController);
  const canCancelGeneration = status === "GENERATING" && Boolean(activeChatController || activeGenerationId !== null);
  $("#chat-cancel").disabled = cancellationInFlight || !(canCancelSearch || canCancelGeneration);
  $("#chat-input").disabled = !["READY","SEARCHING","GENERATING"].includes(status);
  updateSendAvailability();
}

async function refreshChatStatus() {
  try {
    const current = await api("/api/chat/status");
    if (current.active_generation_id != null) activeGenerationId = Number(current.active_generation_id);
    else if (current.status !== "GENERATING") activeGenerationId = null;
    if (activeChatController && current.status === "READY") return;
    setChatStatus(current.status, current.error);
  } catch (error) {
    setChatStatus("ERROR", error.message);
  }
}

function clearEmptyTranscript() {
  const empty = $("#chat-transcript .chat-empty");
  if (empty) empty.remove();
}

function addChatMessage(role, content="") {
  clearEmptyTranscript();
  const message = document.createElement("div");
  message.className = `chat-message ${role}`;
  const label = document.createElement("span");
  label.className = "chat-role";
  label.textContent = role === "user" ? "YOU" : "ASSISTANT";
  const body = document.createElement("span");
  body.className = "chat-body";
  body.textContent = content;
  message.append(label, body);
  $("#chat-transcript").appendChild(message);
  $("#chat-transcript").scrollTop = $("#chat-transcript").scrollHeight;
  return message;
}

function addChatSources(message, sources) {
  if (!message || !Array.isArray(sources) || !sources.length) return;
  const container = document.createElement("div");
  container.className = "chat-sources";
  const title = document.createElement("span");
  title.className = "chat-sources-title";
  title.textContent = "SOURCES";
  container.appendChild(title);
  for (const source of sources.slice(0, 5)) {
    try {
      const url = new URL(source.url);
      if (!["http:", "https:"].includes(url.protocol)) continue;
      const link = document.createElement("a");
      link.className = "chat-source";
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || url.hostname;
      container.appendChild(link);
    } catch (_error) {
      continue;
    }
  }
  if (container.childElementCount > 1) message.appendChild(container);
}

$("#chat-start").addEventListener("click", async () => {
  const button = $("#chat-start");
  await busy(button, async () => {
    try {
      const result = await api("/api/chat/start", "POST");
      setChatStatus(result.status);
      if (result.status === "STARTING") await waitForModelReady();
    } catch (error) {
      setChatStatus("ERROR", error.message);
    }
  });
});

$("#chat-stop").addEventListener("click", async () => {
  const button = $("#chat-stop");
  await busy(button, async () => {
    try {
      const result = await api("/api/chat/stop", "POST");
      setChatStatus(result.status, result.error);
    } catch (error) {
      setChatStatus("ERROR", error.message);
    }
  });
});

$("#chat-clear").addEventListener("click", () => {
  chatHistory.length = 0;
  $("#chat-transcript").innerHTML = '<p class="muted chat-empty">Conversation cleared.</p>';
  $("#chat-metrics").textContent = "";
  $("#chat-input").focus();
});

$("#chat-cancel").addEventListener("click", async () => {
  const controller = activeChatController;
  const generationId = activeGenerationId;
  if (cancellationInFlight || (!controller && generationId === null)) return;
  cancellationInFlight = true;
  $("#chat-cancel").disabled = true;
  if (controller) controller.stopRequested = true;
  try {
    let result;
    if (chatState === "SEARCHING") {
      controller.abort();
      result = await api("/api/chat/cancel", "POST");
    } else {
      result = await api(
        "/api/chat/cancel",
        "POST",
        generationId === null ? undefined : {generation_id:generationId}
      );
      if (controller) controller.abort();
    }
    activeGenerationId = null;
    setChatStatus(result.status, result.error);
  } catch (error) {
    $("#chat-error").textContent = error.message;
  } finally {
    cancellationInFlight = false;
  }
});

$("#chat-input").addEventListener("input", updateSendAvailability);

$("#chat-input").addEventListener("keydown", event => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  if (!$("#chat-input").value.trim() || chatState !== "READY" || activeChatController) return;
  $("#chat-form").requestSubmit();
});

$("#chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const input = $("#chat-input");
  const prompt = input.value.trim();
  if (!prompt || chatState !== "READY" || activeChatController) return;
  const responseStartedAt = performance.now();
  const requestController = new AbortController();
  activeChatController = requestController;
  let requestGenerationId = null;
  input.value = "";
  updateSendAvailability();
  addChatMessage("user", prompt);
  $("#chat-metrics").textContent = "";
  const forceWebSearch = $("#chat-web-search").checked;
  let searchResults = [];
  let searchMetrics = null;
  let route = "LOCAL_QWEN";
  let currentPrompt = prompt;
  let assistant = null;
  let body = null;
  let generated = "";
  let completed = false;
  let requestError = "";
  try {
    setChatStatus("SEARCHING");
    const searchResponse = await fetch("/api/web-search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query: prompt, force_web_search: forceWebSearch}),
      signal: requestController.signal
    });
    const searchData = await searchResponse.json().catch(() => ({}));
    if (!searchResponse.ok) throw new Error(searchData.detail || `Query routing failed with HTTP ${searchResponse.status}.`);
    route = searchData.route || "LOCAL_QWEN";
    searchResults = searchData.results || [];
    if (route === "WEATHER_DIRECT") {
      if (!searchData.direct_answer) throw new Error("Weather lookup returned no answer.");
      assistant = addChatMessage("assistant", searchData.direct_answer);
      generated = searchData.direct_answer;
      completed = true;
      chatHistory.push({role:"user", content:prompt}, {role:"assistant", content:generated});
      addChatSources(assistant, searchResults);
      $("#chat-metrics").textContent = `WEATHER_DIRECT · Open-Meteo fetch ${Number(searchData.fetch_duration_seconds || 0).toFixed(2)}s · Qwen bypassed`;
      return;
    }
    if (["WEATHER_QWEN", "WEB_SEARCH"].includes(route)) {
      if (!searchResults.length || !searchData.context) throw new Error(`${route} failed: no usable evidence was returned.`);
      searchMetrics = {
        searchSeconds: Number(searchData.search_duration_seconds || 0),
        fetchSeconds: Number(searchData.fetch_duration_seconds || 0),
        groundingCharacters: Number(searchData.grounding_character_count || 0)
      };
      currentPrompt = `${prompt}\n\n${searchData.context}`;
    }
    assistant = addChatMessage("assistant");
    assistant.classList.add("streaming");
    body = assistant.querySelector(".chat-body");
    setChatStatus("GENERATING");
    const response = await fetch("/api/chat/completions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: [...chatHistory, {role:"user", content:currentPrompt}], max_tokens:Number($("#chat-max-tokens").value)}),
      signal: requestController.signal
    });
    const generationHeader = response.headers.get("X-Generation-ID");
    if (generationHeader !== null && Number.isInteger(Number(generationHeader))) {
      requestGenerationId = Number(generationHeader);
      activeGenerationId = requestGenerationId;
    }
    if (!response.ok) {
      const failure = await response.json();
      throw new Error(failure.detail || `HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.type === "delta") {
          generated += item.content;
          body.textContent = generated;
          $("#chat-transcript").scrollTop = $("#chat-transcript").scrollHeight;
        } else if (item.type === "metrics") {
          const ttft = item.ttft_seconds == null ? "—" : `${item.ttft_seconds.toFixed(2)}s`;
          const speed = item.decode_tokens_per_second == null ? "—" : `${item.decode_tokens_per_second.toFixed(2)} tok/s`;
          if (searchMetrics) {
            const total = (performance.now() - responseStartedAt) / 1000;
            $("#chat-metrics").textContent = `${route} · Search ${searchMetrics.searchSeconds.toFixed(2)}s · Fetch ${searchMetrics.fetchSeconds.toFixed(2)}s · Grounding ${searchMetrics.groundingCharacters} chars · Qwen TTFT ${ttft} · ${item.completion_tokens ?? "—"} completion tokens · ${speed} · ${total.toFixed(2)}s total`;
          } else {
            $("#chat-metrics").textContent = `TTFT ${ttft} · ${item.completion_tokens ?? "—"} completion tokens · ${speed} · ${item.total_response_seconds.toFixed(2)}s total`;
          }
        } else if (item.type === "done") {
          completed = true;
        } else if (item.type === "error") {
          throw new Error(item.message);
        }
      }
      if (done) break;
    }
    if (!completed) throw new Error("The response stream ended before completion.");
    chatHistory.push({role:"user", content:prompt}, {role:"assistant", content:generated});
    addChatSources(assistant, searchResults);
  } catch (error) {
    const message = (error.name === "AbortError" || requestController.stopRequested) ? "Generation stopped." : error.message;
    requestError = message;
    $("#chat-error").textContent = message;
    if (body && !generated) body.textContent = message;
  } finally {
    if (activeChatController === requestController) activeChatController = null;
    if (activeGenerationId === requestGenerationId) activeGenerationId = null;
    if (assistant) assistant.classList.remove("streaming");
    await refreshChatStatus();
    if (requestError) $("#chat-error").textContent = requestError;
    input.focus();
  }
});

setChatStatus("STOPPED");
refreshChatStatus();
setInterval(refreshChatStatus, 1000);
