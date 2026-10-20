# 11 Multimodal RAG (awareness level)

Let a text question find a picture, and let a model that can see answer from it. Two ways: put pictures and questions in one vector space (CLIP), or describe every picture in words once and use plain text RAG.

```
A. EMBED THE PICTURE
INDEX (once, free):  6 PNGs → CLIP image tower (local) → 512-d vectors → Qdrant `multimodal_medical`
QUERY:  question → CLIP text tower → top 2 images ── best score < 0.25? → refuse in code
                                                  └─ else → gpt-4o-mini LOOKS at both (high detail) → answer

B. DESCRIBE IT IN WORDS
INDEX (once, $0.024):  6 PNGs → gpt-4o-mini writes a caption → bge (naive's embedder) → 6 vectors in memory
QUERY:  question → bge → top 2 captions as TEXT → naive's generate() → answer       (no image sent)
```

Think of two translators who never share notes: one only reads pictures, the other only reads sentences. They were trained together until they wrote their summaries in the same made-up language, so "a chart of hand washing by ward" and a picture of that chart get similar summaries. That's CLIP. Approach B skips the second translator: someone looks at each picture once and writes down what's in it, and from then on it's a normal library.

This follows CLIP (Radford et al., OpenAI, 2021): an image encoder and a text encoder trained on 400 million (image, caption) pairs from the web, so that each image ends up closer to its own caption than to the others. The image encoder is a Vision Transformer (Dosovitskiy et al., 2020): the image is cut into tiles that are treated like words. The "modality gap" we measured is described by Liang et al. (2022, "Mind the Gap"). ColPali (Faysse et al., 2024), which embeds whole page screenshots, is left for the Week 10 deep track.

**There are no images in the corpus**, so this sprint draws its own: 6 synthetic HVMC images whose key fact lives *only in the picture*. It is a demo on 7 sprint-local questions, not a golden-set run: the golden set is text, so pictures can't change its scores.

## How it works

**The images (`make_images.py`).** Drawn with matplotlib. Each one adds a fact the text never states, and never contradicts it: the cap colours are the lab catalog's six, the unit names come from the directory.

| ID | Image | Fact only in the picture |
|---|---|---|
| img01 | Order of draw | Light blue is drawn first |
| img02 | IV-220 front panel | SILENCE is top-left of the keypad |
| img03 | VM-8 main screen | SpO2 is drawn in cyan (HR green, RR yellow) |
| img04 | Hand hygiene chart | 3 South is lowest (71%), the only unit under 90% |
| img05 | Wristband key | Purple = DNR |
| img06 | VM-8 night-mode card | Printed text only: MENU → Display → Night, dims to 30% |

img06 is there on purpose: its answer is printed words, which CLIP is said to read badly.

**How CLIP turns an image into a vector.** The picture is shrunk to 224×224, cut into 7×7 = 49 tiles of 32 px (the "/32" in ViT-B/32), each tile becomes a vector with a position tag, a transformer lets the tiles look at each other, and one summary vector of 512 numbers comes out. The text tower does the same for words, into the same 512 numbers.

**A: embed the picture (`build_index.py`, `retrieve.py`, `pipeline.py`).** FastEmbed's `Qdrant/clip-ViT-B-32-vision` embeds the 6 images into Qdrant; `clip-ViT-B-32-text` embeds the question. The top 2 images go to gpt-4o-mini as images. Two images, not one, because on mm03 the right image came 2nd. If even the best image scores below 0.25, the pipeline refuses without calling the LLM (sprint 09's lesson: an off-topic context is not a reliable "I don't know").

**B: describe it in words (`captions.py`).** gpt-4o-mini writes one caption per image (high detail), told to transcribe every word and to spell out what only the picture shows: which colour means what, what is where, which bar is lowest. Six captions fit in a numpy matrix, so there's no Qdrant collection. Questions are matched with naive's bge embedder and answered by naive's `generate()`, unchanged.

## Run

Start Qdrant first: `docker start qdrant`. From `rag_sprints/`:

```bash
uv run week2/11_multimodal_rag/make_images.py        # draw the 6 images + manifest (free)
uv run week2/11_multimodal_rag/build_index.py        # CLIP-embed them into multimodal_medical (free)
uv run week2/11_multimodal_rag/retrieve.py           # top 3 images per question + the modality gap (free)
uv run week2/11_multimodal_rag/pipeline.py           # A: answer from images (~$0.046)
uv run week2/11_multimodal_rag/captions.py --build   # B: write the 6 captions (~$0.024, then cached)
uv run week2/11_multimodal_rag/captions.py           # B: answer from captions (<$0.001)
```

Set `DETAIL = "low"` in `pipeline.py` to reproduce the low-detail run. The whole sprint cost about $0.12.

## Results (7 sprint-local questions)

### Retrieval: does the question find the right image?

| | CLIP (A) | Captions + bge (B) |
|---|---|---|
| Right image at #1 | 5/6 | **6/6** |
| Right image in top 2 | 6/6 | 6/6 |
| Score of the right image | 0.24–0.34 | 0.59–0.79 |
| Best score on the no-answer question | 0.222 (below every correct #1) | 0.604 (**above** mm04's correct 0.590) |

CLIP's miss is mm03, "what colour is the oxygen saturation number?": it picked the wristband key, the most colourful image, over the monitor. **CLIP matches the topic of a picture, not its details**, because the picture has been shrunk to 224 px before it's read.

**The modality gap.** An image's own title scores **0.35** against that image, but **0.62** against a *different* title (and image vs a different image: 0.43). Text sits near text and images near images, whatever they mean. Training only asked each image to be closer to its own caption *than to other captions*, never to share a neighbourhood with text. This is why every CLIP score is low, and why putting text chunks into the same index would bury the images.

### Answers

| Q | A, low detail | **A, high detail** | **B, captions** |
|---|---|---|---|
| mm01 which tube first | ✅ light blue | ✅ | ✅ |
| mm02 where is the mute button | ❌ "the SILENCE button" | ❌ same | ❌ same |
| mm03 colour of the SpO2 number | ❌ **green** | ✅ cyan | ❌ "blue" |
| mm04 worst hand-washing ward | ✅ 3 South | ❌ "3 North" (once; 4 reruns said 3 South) | ✅ |
| mm05 purple wristband | ❌ **fall risk** | ✅ DNR | ✅ |
| mm06 dim the VM-8 at night | ✅ | ✅ | ✅ |
| mm07 oxygen cylinder (no answer) | ✅ refused in code | ✅ refused in code | ✅ the LLM refused |
| **Correct** | 4/7 | **5/7** | **5/7** |
| Tokens in per question | ~4,900 | ~43,800 | ~490 |
| Cost per question | $0.00075 | $0.0066 | **$0.00008** |
| One-off index cost | free | free | $0.024 |

**In one line:** both approaches answer 5 of 7. Captions find the right image every time and cost about 80× less per question, but whatever the caption got wrong is wrong forever.

## What it gets wrong

- **Low detail misreads, with confidence.** At low detail the image is shrunk to 512 px, and gpt-4o-mini said a purple band means "fall risk" (the row above) and SpO2 is "green" (the HR colour). On a DNR question a confident wrong answer is worse than "I don't know". With only the right image and no second one, it got mm05 right and said "don't know" on mm03, so the extra image also confused it.
- **High detail is better, not reliable.** It fixed both colour questions, at about 9× the tokens. But on one run it read the bar next to 3 South, at temperature 0. Four reruns were right. A text chunk "3 South: 71%" can't be misread that way.
- **A caption error is frozen into the index.** The caption calls SpO2 "blue" and HR "white" (cyan and green), the red STOP button "gray", and 4 North (91%) "approximately 95%". A vision model gets a fresh look at every question; a caption is read once, and every answer after inherits its mistakes.
- **The caption only holds what its prompt asked for.** Ask about something the caption didn't mention (the VM-8 caption never says which corner HR sits in) and the fact is gone. Approach A can look again.
- **No score threshold for captions.** With captions, the no-answer question scored *higher* than a correct one, so refusing depends on the LLM. CLIP's 0.25 cut-off worked here, but it was set from 7 questions.
- **mm02 is a prompt failure in all three runs.** Asked *where* the mute button is, every version answered *which* button ("SILENCE"), though both the image and the caption show it's top-left. The prompt asks for a concise answer, not for answering the question's exact form.

## What it gets right

- **A text question finds a picture.** "Which ward had the worst hand washing compliance" found a bar chart whose bars carry no words about hand washing, using nothing but the shared vector space.
- **CLIP reads large printed text.** The night-mode card (img06) came 1st, though its answer is words only. Short, big text survives the 224 px shrink; a paragraph of small print would not.
- **Captions turn image search into the text RAG we already have.** Same embedder, same `generate()`, same refusals, with clear score margins and no modality gap.
- **Refusing in code worked.** The no-answer question never reached the vision model, so it couldn't describe a cylinder it never saw.

## Ideas not built

- **Mixed index: text chunks + images in one collection.** The modality gap says text chunks would win almost every query. Fixes to try: search the two separately and keep the best of each, or a text-strong joint model like `jina-clip-v1` (FastEmbed has both its towers).
- **Caption + keep the image.** Search captions (cheap, text), then send the original image to the vision model only when the question asks about colour, position or a chart. That's what most production systems do.
- **Check the captions.** Ask a second call "is each statement in this caption true of the image?" and regenerate the ones that fail, much like sprint 07 checks answers.
- **ColPali**: embed whole page screenshots, many vectors per page, matched word by tile (late interaction). FastEmbed has `Qdrant/colmodernvbert` (1 GB) and `colpali-v1.3` (6.5 GB). For real PDFs with charts and tables, in the Week 10 deep track.
- **A modality router**: send "what colour / where / which bar" questions to the images and everything else to text. The plan's ⚡ classifier, for Week 10.
- **Fix the prompt for mm02**: "answer the exact question: if asked where, say where".

## When I'd reach for it

When the answer lives in a picture: charts in reports, scanned forms, device screens, diagrams, photos. **Start with captions**: they're cheap per question, fit the text pipeline you already have, and are easy to check. Keep the images, and send one to a vision model when the question needs fine visual detail, at high detail and with only the image that matters. Use CLIP-style joint embeddings when there are too many images to caption, or when the questions are about what a photo *looks like* rather than what it says. Skip all of it when the documents are text, like our corpus: pictures can't improve answers that are already in the text.

## Files

| File | Purpose |
|---|---|
| `make_images.py` | Draws the 6 synthetic images and writes `images/manifest.jsonl` (each image's hidden fact, used only for checking) |
| `images/` | The 6 PNGs and the manifest |
| `questions_multimodal.jsonl` | 7 sprint-local questions: 6 with answers in images, 1 with no answer |
| `build_index.py` | CLIP image tower → Qdrant `multimodal_medical` |
| `retrieve.py` | `search_images`, the top 3 per question, and the modality gap numbers |
| `pipeline.py` | A: `multimodal_pipeline` (top 2 images → gpt-4o-mini, refusal below 0.25), runs the 7 questions |
| `captions.py` | B: writes the captions (`--build`), `caption_pipeline` (bge over captions → naive's `generate()`) |
| `captions_medical.json` | The cached captions with their token counts |
