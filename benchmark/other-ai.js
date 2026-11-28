// The AI that Pyinsect is compared with. CLIP runs on this device. OpenRouter is a chatbot AI on the internet.
// The OpenRouter key is only saved (in this browser) if the grown-up ticks "Remember my key".

window.OtherAI = (() => {
const OPENROUTER = "https://openrouter.ai/api/v1";
const OPENROUTER_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free";   // ":free" models cost $0, but limit how often you can ask
// The free model costs $0. To show what asking could cost, we estimate with the prices (dollars per
// million tokens) of the paid, text-only NVIDIA Nemotron 3 Nano 30B A3B.
const PRICE_IN = 0.06, PRICE_OUT = 0.24;
const MAX_DOLLARS = 1;   // stop asking after about this much, each time the page is open
const CLIP_MODEL = "Xenova/clip-vit-base-patch32";
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

// The 3 best of a list of [species, score].
const best3 = (scores) => scores.sort((a, b) => b[1] - a[1]).slice(0, 3)
  .map(([species, confidence]) => ({ species, confidence }));

/* ---------- CLIP: compares the photo with the words for each insect ---------- */
function clip(species) {
  let ready = null;
  async function load(status) {
    status("Downloading CLIP. The first time is big, so use Wi-Fi…");
    const t = await import(TRANSFORMERS);
    const options = { dtype: "q8" };   // the small version
    const [tokenizer, processor, textModel, visionModel] = await Promise.all([
      t.AutoTokenizer.from_pretrained(CLIP_MODEL),
      t.AutoProcessor.from_pretrained(CLIP_MODEL),
      t.CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL, options),
      t.CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, options),
    ]);
    // The words are the same for every photo, so turn them into numbers only once.
    const words = [];
    for (let i = 0; i < species.length; i += 32) {
      status(`CLIP is reading the insect names… ${i} of ${species.length}`);
      const texts = species.slice(i, i + 32).map((s) => `a photo of ${commonName(s)}, ${sciName(s)}`);
      const { text_embeds } = await textModel(tokenizer(texts, { padding: true, truncation: true }));
      words.push(...text_embeds.normalize(2, -1).tolist());
    }
    return { t, processor, visionModel, words };
  }
  const getReady = (status) => {
    ready = ready || load(status).catch((e) => { ready = null; throw e; });
    return ready;
  };
  return {
    name: "CLIP",
    kind: "(generalist) runs on this device",
    getReady,
    async ask(file) {
      const { t, processor, visionModel, words } = await getReady(() => {});
      const start = performance.now();
      const image = await t.RawImage.fromBlob(file);
      const { image_embeds } = await visionModel(await processor(image));
      const [photo] = image_embeds.normalize(2, -1).tolist();
      const logits = words.map((w) => 100 * w.reduce((sum, v, i) => sum + v * photo[i], 0));
      const most = Math.max(...logits), parts = logits.map((v) => Math.exp(v - most));
      const total = parts.reduce((a, b) => a + b, 0);
      const top3 = best3(species.map((s, i) => [s, (100 * parts[i]) / total]));
      return { top3, ms: performance.now() - start, cost: 0, tokens: 0 };
    },
  };
}

/* ---------- OpenRouter ---------- */
// True if OpenRouter accepts the key, false if it doesn't. Throws if OpenRouter can't be reached.
async function keyWorks(key) {
  const res = await fetch(`${OPENROUTER}/key`, { headers: { Authorization: `Bearer ${key}` } });
  if (res.status === 401 || res.status === 403) return false;
  if (!res.ok) throw new Error(`OpenRouter said ${res.status}`);
  return true;
}

// A new, smaller JPEG copy. This also leaves out hidden details, like where the photo was taken.
async function smallJpeg(file) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(bmp.width * scale));
  c.height = Math.max(1, Math.round(bmp.height * scale));
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c.toDataURL("image/jpeg", 0.9);
}

// Reads {"top3": [{"id": ..., "confidence": ...}]} out of the reply. A reply we can't read counts as wrong.
function readAnswer(text, species) {
  try {
    return JSON.parse(text.match(/\{.*\}/s)[0]).top3.slice(0, 3).map((g) => ({
      species: species.includes(String(g.id).trim()) ? String(g.id).trim() : null,
      confidence: Number(g.confidence) || 0,
    }));
  } catch { return []; }
}

// Only free models: the ID ends in ":free" and every listed price is 0.
const isFree = (m) => m.id.endsWith(":free") && Object.values(m.pricing || {}).every((p) => Number(p) === 0);
let freeList = null, listedAt = 0;
// The free models that can read pictures and answer in words, from OpenRouter's public list (no key needed).
async function freeModels() {
  if (freeList && Date.now() - listedAt < 600000) return freeList;   // ask again after 10 minutes
  const res = await fetch(`${OPENROUTER}/models`, { signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw new Error(`OpenRouter said ${res.status}`);
  freeList = (await res.json()).data
    .filter((m) => isFree(m) && m.architecture?.input_modalities?.includes("image")
      && m.architecture?.output_modalities?.includes("text"))
    .map((m) => ({ id: m.id, name: m.name }))
    .sort((a, b) => a.name.localeCompare(b.name));
  listedAt = Date.now();
  return freeList;
}

// Checked before every photo, so a model that costs money is never asked.
async function mustBeFree(model) {
  if (!model.endsWith(":free")) throw new Error("not a free model");
  let list;
  try { list = await freeModels(); }
  catch (e) { console.warn("couldn't load the model list:", e); return; }   // ":free" IDs are still free
  if (!list.some((m) => m.id === model)) throw new Error("this model isn't free anymore");
}

function openRouter(species, key, startModel) {
  let model = startModel?.endsWith(":free") ? startModel : OPENROUTER_MODEL;
  let spent = 0;
  const listing = species.map((s) => `${s} = ${commonName(s)}`).join("\n");
  const prompt = "Name the insect (or the insect damage) in this photo. Choose only from this list of ids:\n"
    + `${listing}\n\nReply with only JSON, best guess first: `
    + '{"top3": [{"id": "<id from the list>", "confidence": <0 to 100>}]}';
  return {
    name: "OpenRouter",
    get kind() { return `(generalist) ${model}`; },
    get model() { return model; },
    setModel(id) { if (id.endsWith(":free")) model = id; },
    freeModels,
    estimate: true,   // its cost is an estimate, not money spent
    async getReady() {},
    async ask(file) {
      if (spent >= MAX_DOLLARS) throw new Error("spending limit");
      await mustBeFree(model);
      const image = await smallJpeg(file);
      const start = performance.now();
      const res = await fetch(`${OPENROUTER}/responses`, {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        signal: AbortSignal.timeout(180000),
        body: JSON.stringify({ model, input: [{ role: "user", content: [
          { type: "input_text", text: prompt },
          { type: "input_image", image_url: image, detail: "auto" },
        ] }] }),
      });
      if (!res.ok) throw new Error(`OpenRouter said ${res.status}`);
      const reply = await res.json();
      const ms = performance.now() - start;
      spent += Number(reply.usage?.cost) || 0;   // real money, for the spending limit
      const cost = ((Number(reply.usage?.input_tokens) || 0) * PRICE_IN
        + (Number(reply.usage?.output_tokens) || 0) * PRICE_OUT) / 1e6;
      const text = (reply.output || []).filter((item) => item.type === "message")
        .flatMap((item) => item.content || []).filter((part) => part.type === "output_text")
        .map((part) => part.text).join("");
      return { top3: readAnswer(text, species), ms, cost, tokens: Number(reply.usage?.total_tokens) || 0 };
    },
    overLimit: () => spent >= MAX_DOLLARS,
  };
}

return { clip, openRouter, keyWorks };
})();
