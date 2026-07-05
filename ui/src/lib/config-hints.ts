export const configHints = {
  provider:
    "Local AirLLM a gepeden futtatja a modellt. OpenAI-compatible kulso API-t hasznal (pl. Ollama, LM Studio, felho).",
  providerPreset: "Elore beallitott kulso szolgaltato sablon. Kitolti a Base URL-t es a model nevet.",
  baseUrl: "A kulso API vegpontja, altalaban /v1 vegzodessel (pl. https://api.example.com/v1).",
  externalModel: "A kulso szolgaltato altal vart modellazonosito neve.",
  externalTimeout: "Maximalis varakozasi ido masodpercben a kulso API valaszara.",
  externalApiKey:
    "API kulcs, ha a szolgaltato keri. Csak a kereshez hasznaljuk, nem taroljuk el a szerveren.",
  preset: "Gyors modellvalasztas ismert Hugging Face modellekhez. Egyedi ID-hez valaszd az Egyedi opciot.",
  modelId:
    "Hugging Face modell ID vagy helyi utvonal (pl. Qwen/Qwen2.5-Coder-3B-Instruct). Ezt tolti be az AirLLM.",
  device:
    "Hol fusson a modell: auto a hardver alapjan valaszt, cuda:0 NVIDIA GPU-n, mps Apple Silicon/Metal rendszeren, cpu processzoron.",
  dtype:
    "Szamitas pontossaga: float16/bfloat16 gyorsabb GPU-n, float32 pontosabb, auto a rendszer ajanlja.",
  compression:
    "Modell tomoritese memoriaba: 4bit/8bit kevesebb RAM-ot/GPU memoriát hasznal, none teljes pontossag.",
  loadMode:
    "Auto valaszt. GPU/MPS rezidens a leggyorsabb, ha belefer. CPU+GPU hybrid CUDA-n megosztja a modell retegeit VRAM es RAM kozott. AirLLM streaming nagyon nagy modellekhez lassabb fallback.",
  prefetching:
    "Kovetkezo modell retegek elore betoltese CPU/GPU kozott. Gyorsitja a generalast nagy modelleknél.",
  maxSeqLen:
    "Maximalis kontextus hossz tokenben. Nagyobb ertek tobb szoveget enged, de tobb memoriat igenyel.",
  cleanupInterval:
    "Hany forward lepes utan fusson memoria tisztitas. Magasabb ertek = kevesebb szunet, tobb memoria.",
  prefetchWorkers:
    "Parhuzamos szalak szama a retegek elore betoltesere (1-4). Tobb worker gyorsabb lehet NVMe-n.",
  layerCache:
    "Hova mentse az AirLLM a szetbontott modell retegeit. Uresen a Hugging Face cache mappat hasznalja.",
  hfToken:
    "Hugging Face token gated vagy privat modellek letolteséhez. Uresen hagyhato nyilvanos modelleknel.",
  profiling: "Reszletes teljesitmenymeres bekapcsolasa betolteskor. Hibakereseshez, lassabb lehet.",
  deleteOriginal:
    "Az eredeti letoltott modell torlese a cache-bol a szetbontas utan. Helyet szabadit fel.",
  reinitForward:
    "Kompatibilitasi mod: minden forward pass elott ujra inicializalja a modellt. Lassu, csak hiba eseten.",
  agentObjective: "Mit csinaljon a coding agent: elemzes, javaslat, kodterv vagy refaktor feladat.",
  agentWorkspace: "Melyik mappa fajljait olvassa be kontextusnak. Uresen a projekt gyokeret hasznalja.",
  agentContextChars: "Maximalis karakterszam a beolvasott fajlokbol. Nagyobb = tobb kontextus, lassabb.",
  inputMax: "Bemeneti szoveg maximalis hossza tokenben az inference elott.",
  newTokens: "Hany uj tokent generaljon maximum a modell egy valaszban.",
  temperature: "Kreativitas: alacsony = kiszamithato, magas = valtozatosabb valaszok (0-2).",
  topP: "Nucleus sampling: a valoszinusegi tomeg top p reszet tartja meg (0.05-1).",
  topK: "Csak a legvaloszinubb K token kozul valaszt. 0 = kikapcsolva.",
  repeatPenalty: "Ismetlodo szavak buntetese. 1 felett csokkenti az ismetlest.",
  autoload: "Ha nincs betoltott modell, automatikusan betolti a kivalasztott modellt generalas elott.",
  kvCache: "KV cache hasznalata: gyorsitja a tobb lepeses generalast, tobb memoriat hasznal.",
  chatTemplate: "A modell chat sablonjanak alkalmazasa a bemeneti uzenetekre (ajanlott beszelgeteshez).",
} as const
