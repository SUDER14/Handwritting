# PROJECT: AI-Based Personalized Handwriting Generation from a Small Handwritten Sample

You are an expert AI/ML engineer, computer vision engineer, Python developer, and research-oriented software architect.

I want you to BUILD THE PROJECT, not just explain it.

You have access to my development environment through Claude Code. Work directly on the project files, create the required folder structure, install dependencies where appropriate, write the code, run it, test it, debug it, and continuously improve it.

Do not stop at giving me code snippets. Actually implement the complete working project.

---

# 1. CORE IDEA

I want to build an AI system that can learn a person's handwriting style from a VERY SMALL handwritten sample, initially even a single word such as:

"quick"

The system should analyze the handwritten word and learn:

* Letter shapes
* Stroke characteristics
* Character proportions
* Character height
* Character width
* Slant
* Curvature
* Stroke thickness
* Spacing
* Baseline
* Overall handwriting style

The handwritten word contains only some characters.

For example:

quick

contains:

q, u, i, c, k

The system should identify these observed characters and extract both:

1. Character-specific information
2. General writer-specific handwriting style

Then the system should use AI/ML to infer/generate the missing characters of the alphabet in the same handwriting style.

For example:

Observed:

q u i c k

Generated:

a b d e f g h j l m n o p r s t v w x y z

The ultimate goal is to generate a complete personalized handwriting alphabet.

---

# 2. IMPORTANT CONCEPT

Do NOT simply create a normal font by copying the five letters.

The research/AI objective is:

PARTIAL HANDWRITING SAMPLE
↓
HANDWRITING FEATURE EXTRACTION
↓
WRITER STYLE EMBEDDING
↓
CONDITIONAL GENERATION
↓
MISSING CHARACTER GENERATION
↓
PERSONALIZED ALPHABET
↓
ARBITRARY TEXT GENERATION

The model should learn the STYLE of the writer and use that style to generate unseen characters.

The generated character should look as if it belongs to the same person's handwriting.

---

# 3. FIRST VERSION / MVP

Do NOT begin by building an enormous diffusion model or an unnecessarily complicated architecture.

Build a working MVP first.

The MVP should support:

### Input

A handwritten image uploaded by the user.

Example:

quick

The user should be able to upload:

* PNG
* JPG
* JPEG

The image may contain a handwritten word or short sentence.

---

# 4. INPUT PROCESSING

Implement a robust preprocessing pipeline.

The pipeline should include:

1. Image loading
2. Grayscale conversion
3. Noise removal
4. Contrast normalization
5. Thresholding / binarization
6. Background removal where possible
7. Deskewing
8. Cropping
9. Connected-component analysis
10. Character segmentation
11. Character normalization
12. Baseline estimation
13. Character bounding-box extraction

The preprocessing system should be modular so that it can later be improved.

Use OpenCV and appropriate Python libraries.

---

# 5. CHARACTER SEGMENTATION

The system should attempt to identify individual handwritten characters from the input.

For example:

Input:

quick

should approximately become:

[q] [u] [i] [c] [k]

However, handwriting can contain connected letters.

Therefore, do not assume simple whitespace segmentation will always work.

Create a modular segmentation pipeline that can later support:

* Connected components
* Projection profiles
* Contour analysis
* Vertical projection
* Morphological operations

For the MVP, use the most reliable practical approach.

Document limitations clearly.

---

# 6. CHARACTER IDENTIFICATION

The system needs to know which character each segmented glyph represents.

For the MVP, support:

* Lowercase English letters
* Uppercase English letters
* Digits

But prioritize lowercase a-z first.

Design this so that the recognition component can later be replaced by a trained OCR/character classification model.

Possible approaches:

* CNN classifier
* pretrained OCR
* template matching for prototype
* lightweight ML classifier

Do not hard-code the final solution unnecessarily.

---

# 7. HANDWRITING STYLE EXTRACTION

This is the most important part of the project.

Create a STYLE ENCODER.

The style encoder should extract writer-specific characteristics from the handwritten sample.

Start with a practical neural architecture.

Possible architecture:

Input glyph images
↓
CNN feature extractor
↓
Character embeddings
↓
Feature aggregation
↓
Style embedding
↓
Fixed-dimensional vector

For example:

style_embedding = [x1, x2, x3, ..., xn]

The exact dimensionality should be selected based on the architecture.

The style embedding should capture information such as:

* Slant
* Stroke width
* Character proportions
* Roundness
* Curvature
* Spacing tendencies
* Height
* Width
* Baseline characteristics
* Overall visual appearance

Separate CHARACTER CONTENT from WRITER STYLE as much as reasonably possible.

This separation is important.

---

# 8. HANDWRITING STYLE VS CHARACTER IDENTITY

The architecture should conceptually learn:

IMAGE = CHARACTER CONTENT + WRITER STYLE

For example:

Writer A:

"a"

Writer B:

"a"

The character identity is the same:

"a"

But the style is different.

Likewise:

Writer A:

"a", "b", "c", "q", "k"

should share a common writer-style representation.

Design the system so that character identity and handwriting style are represented separately.

Use embeddings where appropriate.

---

# 9. GENERATING MISSING LETTERS

This is the core generative component.

Suppose the user provides:

quick

Observed letters:

q u i c k

The system should use:

* Character identity
* Learned writer style embedding
* Observed handwriting examples

to generate:

a b d e f g h j l m n o p r s t v w x y z

The generator should be conditioned on:

character identity + handwriting style

Conceptually:

GeneratedGlyph = Generator(CharacterEmbedding, StyleEmbedding)

For example:

Generator("a", UserStyle)
↓
personalized handwritten "a"

Generator("b", UserStyle)
↓
personalized handwritten "b"

etc.

---

# 10. IMPORTANT: FEW-SHOT LEARNING

The main research direction is FEW-SHOT PERSONALIZED HANDWRITING GENERATION.

The user may provide only a small number of characters.

Do not require all 26 characters during inference.

The system should attempt to infer the general writing style from whatever characters are available.

For example:

Input 1:

quick

Input 2:

hello

Input 3:

machine learning

The system should combine the available evidence to improve the style embedding.

The architecture should support increasing the amount of reference handwriting without changing the fundamental pipeline.

---

# 11. TRAINING STRATEGY

Do NOT assume that we have enough data to train a large generative model from scratch.

First investigate suitable public handwriting datasets.

Consider datasets such as:

* IAM Handwriting Database
* EMNIST
* CVL Handwriting Dataset
* other legally usable handwriting datasets

Use a dataset that allows us to train/validate the general handwriting representation.

The important concept is:

TRAINING:

Many writers
↓
Learn handwriting style representation
↓
Learn character/style relationship

INFERENCE:

New writer
↓
Only a few handwritten characters/words
↓
Extract new writer's style embedding
↓
Generate unseen characters

This is much more appropriate than training a completely new model every time a new user uploads "quick".

---

# 12. MODEL DEVELOPMENT STRATEGY

Implement the project in stages.

## Stage 1

Build a non-generative baseline.

Use extracted handwriting features to create a personalized character representation.

Verify that the system can distinguish different handwriting styles.

## Stage 2

Build the neural style encoder.

## Stage 3

Build character embeddings.

## Stage 4

Build a conditional glyph generator.

## Stage 5

Generate missing letters.

## Stage 6

Generate complete alphabet.

## Stage 7

Generate arbitrary words/sentences.

## Stage 8

Export the generated handwriting.

Do not skip directly to Stage 8 without validating earlier stages.

---

# 13. GENERATION OUTPUT

The generated alphabet should be stored as individual glyphs.

For example:

generated_alphabet/

```
lowercase/

    a.png
    b.png
    c.png
    ...
    z.png

uppercase/

    A.png
    B.png
    ...
    Z.png

digits/

    0.png
    ...
    9.png
```

Store metadata as well.

Example:

glyph_metadata.json

containing:

* character
* bounding box
* width
* height
* baseline
* style embedding reference
* generation information

---

# 14. TEXT-TO-HANDWRITING ENGINE

Once the personalized alphabet exists, implement a text renderer.

Input:

"Artificial Intelligence"

Output:

A generated handwritten image that looks like the learned user's handwriting.

The renderer should support:

* Character spacing
* Word spacing
* Baseline
* Character width
* Character height
* Natural variation
* Slight positional variation
* Optional rotation/slant variation

Do NOT make every occurrence of a character pixel-identical.

Real handwriting has natural variation.

If the model generates multiple variants of a character, use them intelligently.

For example:

a_variant_1.png
a_variant_2.png
a_variant_3.png

Then the renderer can select different variants.

---

# 15. SENTENCE GENERATION

The system should eventually support:

Input:

"Artificial Intelligence is changing the world."

Output:

An image containing the same sentence rendered in the learned handwriting style.

Maintain:

* Natural spacing
* Baseline
* Line height
* Character proportions
* Word spacing

---

# 16. USER INTERFACE

Build a simple local web application.

Use:

Python backend

and preferably:

Streamlit

for the first MVP.

The UI should contain:

---

PERSONALIZED HANDWRITING GENERATOR

1. Upload handwriting sample

[ Upload Image ]

2. Preview uploaded handwriting

3. Detected characters

q u i c k

4. Handwriting style analysis

Show:

* Slant
* Stroke width
* Average character height
* Average character width
* Spacing
* Style embedding dimensions

5. Generate Alphabet

[ Generate ]

6. Preview generated alphabet

7. Enter text

[ Artificial Intelligence is changing the world. ]

8. Generate Handwriting

[ Generate ]

9. Output image

---

---

# 17. PROJECT STRUCTURE

Create a professional project structure similar to:

handwriting-ai/

```
README.md

requirements.txt

.gitignore

config/

    config.yaml

data/

    raw/

    processed/

    samples/

models/

    checkpoints/

notebooks/

src/

    preprocessing/

    segmentation/

    recognition/

    features/

    style_encoder/

    generator/

    renderer/

    evaluation/

    utils/

app/

    app.py

tests/

outputs/

    alphabets/

    generated_text/

scripts/

    preprocess_dataset.py

    train_style_encoder.py

    train_generator.py

    generate_alphabet.py

    generate_text.py

docs/

    architecture.md

    methodology.md

    experiments.md

    limitations.md

    future_work.md
```

---

# 18. CODE QUALITY

Write production-quality Python.

Use:

* Type hints
* Docstrings
* Modular functions
* Classes where appropriate
* Configuration files
* Logging
* Error handling

Avoid putting the entire project into one Python file.

Avoid unnecessary complexity.

---

# 19. TESTING

Create unit tests for:

* Image preprocessing
* Segmentation
* Character normalization
* Style feature extraction
* Style encoder
* Generator
* Alphabet generation
* Text rendering

Also create at least one end-to-end test:

handwriting image
↓
preprocessing
↓
segmentation
↓
style extraction
↓
alphabet generation
↓
text generation
↓
output image

---

# 20. EVALUATION

Do not only visually inspect the result.

Implement quantitative evaluation where practical.

Evaluate:

### Character similarity

Compare generated glyphs against target writer samples when ground truth is available.

Possible metrics:

* SSIM
* PSNR
* LPIPS

### Recognition accuracy

Run OCR on generated handwriting.

Measure:

Character Recognition Accuracy

and:

Word Recognition Accuracy

### Writer similarity

Investigate whether a writer-identification model can recognize the generated sample as belonging to the correct writer.

This is particularly important.

A good generated sample should:

1. Contain the correct character.
2. Look handwritten.
3. Preserve the writer's style.

### Human evaluation

Design an experiment where people rate:

* Similarity to reference handwriting
* Naturalness
* Readability

---

# 21. BASELINE MODELS

Implement at least one baseline.

For example:

BASELINE:

Simple glyph extraction + nearest style transformation

versus:

PROPOSED:

Style Encoder + Conditional Generator

Compare them.

This will help make the project suitable for an academic/research project.

---

# 22. EXPERIMENTAL QUESTION

The main research question should be:

"How effectively can a model generate unseen handwritten characters in a writer's style when provided with only a small number of handwritten reference characters?"

Create experiments based on:

1 reference word

3 reference words

5 reference words

short sentence

multiple sentences

Compare performance as reference data increases.

---

# 23. IMPORTANT TECHNICAL REQUIREMENT

Do not train a massive model blindly.

First inspect the available machine:

* CPU
* GPU
* RAM
* VRAM
* Python version
* CUDA availability

Use:

PyTorch

if suitable.

If a GPU is available, use it.

If not, build a lightweight CPU-compatible prototype.

The project MUST remain runnable on a normal development machine.

---

# 24. MODEL SELECTION

You have permission to choose the most appropriate architecture.

Consider:

* CNN
* Siamese Network
* Autoencoder
* Variational Autoencoder
* Conditional VAE
* CNN + Transformer
* GAN
* Diffusion model

However, choose the simplest architecture that can demonstrate the research idea effectively.

Do not use a complex model merely because it sounds impressive.

Document why the chosen architecture was selected.

---

# 25. IMPORTANT MVP FALLBACK

If full AI generation cannot initially produce high-quality unseen characters, implement a hybrid architecture:

Observed character:
→ directly preserve real glyph

Unobserved character:
→ AI-generated glyph

Style parameters:
→ extracted from all available handwriting

This means if the input contains:

quick

then:

q → real observed q
u → real observed u
i → real observed i
c → real observed c
k → real observed k

while:

a,b,d,e,f,...
→ generated using the learned style

This hybrid approach is acceptable and should be implemented if it produces better practical results.

---

# 26. VERY IMPORTANT: DO NOT FAKE AI

Do not create a project where the "AI model" is merely:

* random image transformations
* rotating letters
* changing font size
* changing font thickness
* applying OpenCV filters

Those techniques can be used as preprocessing or baseline methods, but the final proposed system should contain a genuine ML component.

The system should learn a meaningful representation of handwriting style.

---

# 27. DEVELOPMENT WORKFLOW

Follow this exact workflow.

STEP 1:
Inspect the existing project directory.

STEP 2:
Inspect the development environment.

STEP 3:
Determine available Python/PyTorch/GPU resources.

STEP 4:
Create the project structure.

STEP 5:
Create a requirements.txt.

STEP 6:
Implement preprocessing.

STEP 7:
Implement segmentation.

STEP 8:
Implement character recognition.

STEP 9:
Implement baseline style extraction.

STEP 10:
Test the pipeline using a small handwritten sample.

STEP 11:
Select an appropriate public dataset.

STEP 12:
Download/use the dataset only if legally and technically appropriate.

STEP 13:
Implement training pipeline.

STEP 14:
Train a small initial model.

STEP 15:
Evaluate it.

STEP 16:
Implement conditional generation.

STEP 17:
Generate unseen alphabet characters.

STEP 18:
Implement text rendering.

STEP 19:
Create Streamlit UI.

STEP 20:
Run end-to-end tests.

STEP 21:
Fix errors.

STEP 22:
Improve output quality.

STEP 23:
Document the architecture and methodology.

---

# 28. DO NOT ASK ME FOR EVERY SMALL DECISION

You have permission to make reasonable engineering decisions yourself.

If there are multiple technically valid choices, choose the one that:

1. Is easiest to implement correctly
2. Can run locally
3. Has academic value
4. Can be evaluated quantitatively
5. Can later be upgraded

Only ask me a question if the decision genuinely requires information that cannot be inferred from the environment.

Otherwise proceed.

---

# 29. DO NOT CLAIM SOMETHING WORKS WITHOUT TESTING IT

After implementing each major component:

RUN IT.

If it fails:

DEBUG IT.

If the model produces poor output:

ANALYZE WHY.

Then improve it.

Do not simply tell me:

"Run this command."

Actually run the command when possible.

---

# 30. FINAL APPLICATION BEHAVIOR

The final application should work approximately like this:

USER:

Uploads:

[ handwritten image containing "quick" ]

SYSTEM:

Detects:

q u i c k

↓

Extracts handwriting characteristics

↓

Creates writer style embedding

↓

Generates missing characters

↓

Creates personalized alphabet

↓

Displays:

A B C D E F G ... Z

a b c d e f g ... z

↓

User enters:

"Hello, my name is Sudarshan."

↓

System generates:

[handwritten image]

in the learned handwriting style.

---

# 31. FUTURE EXTENSIONS

Architect the code so that these can be added later:

* Multiple handwriting samples
* Full paragraph generation
* Handwriting on ruled paper
* Handwriting on blank paper
* Different pen styles
* Blue/black ink
* Handwriting pressure simulation
* Stroke trajectory generation
* Cursive handwriting
* Multiple variants of each character
* Personalized punctuation
* Numbers
* Mathematical symbols
* Export to PNG
* Export to SVG
* Export to PDF
* Custom TTF/OTF font generation
* Microsoft Word integration
* Notepad integration
* Windows virtual printer
* Handwritten document generation

Do not implement all of these now.

Design the architecture so they can be added later.

---

# 32. ACADEMIC DOCUMENTATION

Create:

docs/architecture.md

docs/methodology.md

docs/experiments.md

docs/limitations.md

docs/future_work.md

Also create a detailed README.md containing:

* Project title
* Problem statement
* Motivation
* Existing approaches
* Research gap
* Proposed methodology
* Architecture
* Technologies
* Installation
* Usage
* Training
* Evaluation
* Results
* Limitations
* Future work

---

# 33. RESEARCH HONESTY

Clearly distinguish between:

EXISTING RESEARCH

and

OUR IMPLEMENTATION

Do not claim that the overall concept is novel if similar handwriting synthesis systems already exist.

Instead, identify potential novelty around:

* Few-shot personalization
* Small reference sample size
* Partial alphabet inference
* Writer-style embedding
* Hybrid observed/unseen character generation
* Quantitative evaluation of how much reference handwriting is needed

---

# 34. FIRST TASK

Do NOT immediately write the entire system blindly.

First inspect the current environment and project directory.

Then create:

1. Architecture plan
2. Project structure
3. Initial implementation plan
4. Dependency list

Then begin implementing the MVP.

Keep a development log in:

docs/development_log.md

After each major milestone, update it with:

* What was implemented
* What was tested
* What worked
* What failed
* What was changed
* Next step

---

# 35. SUCCESS CRITERIA

The project is considered successful when I can:

1. Start the application locally.
2. Upload a handwritten word such as "quick".
3. See the detected characters.
4. See extracted handwriting/style information.
5. Generate the missing alphabet.
6. See the generated alphabet.
7. Enter arbitrary English text.
8. Generate that text in the learned handwriting style.
9. Save the generated handwriting as an image.
10. Repeat the process with another person's handwriting and obtain a visibly different style.

The system should NOT simply use a pre-existing computer font as the final output.

The final output must be based on the user's handwriting style.

---

# START NOW

Begin by inspecting the environment and existing files.

Then implement the project incrementally.

Do not merely provide instructions to me.

ACT AS THE DEVELOPER AND BUILD THE PROJECT.
