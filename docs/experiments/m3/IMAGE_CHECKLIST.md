# M3 experiment: image checklist

You need **51 photos**: **37 test** photos (one per scenario, two for the 7 "both" scenarios) and **14 development** photos (used only to check and tune the posture and skin rules, never to report results).

Save them in `data/experiments/m3/images/test/` and `.../dev/` (this folder is git-ignored; never commit or publish the photos). Record every photo in `data/experiments/m3/image_sources.csv`.

## 1. Every photo must

- [ ] show **one adult**, facing the camera (front view, head turned no more than ~30°);
- [ ] show the **face clearly** (no sunglasses, mask or heavy shadow; a hand on the forehead is fine in head-pain photos);
- [ ] show the **upper body down to at least the hips**, with **both wrists visible** (needed to measure posture) — except the `_face` photos of the "both" cases, which may be face close-ups;
- [ ] be **natural colour**, no filters, black-and-white, heavy make-up effects or strong coloured lighting (these fool the skin check);
- [ ] be at least **640 px** on the shorter side, JPG or PNG;
- [ ] be a **real photo** (no AI-generated, drawn or watermarked images);
- [ ] come from an **allowed source** (section 3) with its page link recorded.

Try to include **Asian / Thai-looking** people where you can (the study's target population), plus a mix of men and women.

## 2. What to collect

### Test set — 37 photos for 30 scenarios

| Category | Count | File names | What it shows |
| --- | --- | --- | --- |
| Neither | 8 | `neither_01.jpg` … `neither_08.jpg` | Calm person, normal skin colour, arms relaxed (standing, or sitting with hands **not** on the lap/stomach) |
| Posture: chest | 3 | `chest_01.jpg` … | Hand or fist pressed on the **middle of the chest**, normal skin |
| Posture: abdomen | 3 | `abdomen_01.jpg` … | Hand(s) holding the **stomach** (around the navel or upper stomach), normal skin |
| Posture: head | 2 | `head_01.jpg` … | Hand on the **forehead or temple** (headache), normal skin |
| Flushing only | 7 | `flush_01.jpg` … | Clearly **red / flushed face** (cheeks and/or forehead), arms relaxed |
| Both: flushed + head | 3 **pairs** | `both_head_01_body.jpg` + `both_head_01_face.jpg` … | **body** photo: hand on forehead/temple · **face** photo: a different person's clearly red face |
| Both: flushed + chest | 2 **pairs** | `both_chest_01_body.jpg` + `both_chest_01_face.jpg` … | **body**: hand on chest · **face**: red face |
| Both: flushed + abdomen | 2 **pairs** | `both_abdomen_01_body.jpg` + `both_abdomen_01_face.jpg` … | **body**: holding stomach · **face**: red face |

**Why two photos for "both":** photos showing a red face *and* a pain posture together are very hard to find. The camera part gives two separate results (posture, skin), so for these 7 cases the posture is read from the `_body` photo and the skin from the `_face` photo. In `image_sources.csv`, give the body photo its real `true_posture` (and its real `true_flushing`), and the face photo `true_posture = none`, `true_flushing = yes`. Use 7 body photos and 7 face photos that are **not** used anywhere else.

### Development set — 14 photos

| Category | Count | File names | Why |
| --- | --- | --- | --- |
| Chest | 2 | `dev_chest_01.jpg` … | Check the chest rule |
| Abdomen | 2 | `dev_abdomen_01.jpg` … | Check the abdomen rule |
| Head | 2 | `dev_head_01.jpg` … | Check the headache rule |
| Thinking pose (hand at chin) | 2 | `dev_thinking_01.jpg` … | Must **not** count as headache |
| Sitting, hands resting on lap | 2 | `dev_lap_01.jpg` … | Must **not** count as abdominal pain |
| Arms crossed | 1 | `dev_crossed_01.jpg` | Must **not** count as chest pain |
| Flushed face | 2 | `dev_flush_01.jpg` … | Check the skin patches and threshold |
| Neutral | 1 | `dev_neither_01.jpg` | Baseline |

## 3. Where to download (allowed sources)

| Source | Licence (checked) | Use it for |
| --- | --- | --- |
| [Pexels](https://www.pexels.com/license/) | Free use and modification, no attribution required; identifiable people must not be shown in a bad light; no redistribution on other stock sites | Postures, neutral, some flushed/fever photos |
| [Unsplash](https://unsplash.com/license) | Free use and modification, no attribution required; no compiling into a competing service | Postures, neutral |
| [CDC Public Health Image Library (PHIL)](https://phil.cdc.gov/) | Many images are **public domain** — check the "Copyright Restrictions" line on each image page | Flushed faces (e.g. sunburn, fever) |
| [Wikimedia Commons](https://commons.wikimedia.org/) | Licence differs per file (CC0, CC BY, CC BY-SA…) — copy it from the file page | Any category, if the licence allows reuse |
| [Openverse](https://openverse.org/) | Search engine for openly licensed images; licence shown per image | Finding Commons / CC photos |

**Search words to try:** "chest pain", "heart attack", "stomach ache", "abdominal pain", "headache", "migraine", "fever", "flushed face", "red face", "sunburn face", "hot flash", "Asian man headache", "Asian woman stomach ache", "thinking pose", "sitting hands on lap", "arms crossed".

### Do NOT use

- **Flickr** — the CNN's FFHQ training data was built from Flickr photos (risk of the same photo).
- **Kaggle or Roboflow** image datasets, **FairFace**, **FFHQ** — the CNN was trained/tested on these.
- **Dermatology atlases** (e.g. DermNet) — likely overlap with the skin-disease training data.
- Stock sites that require payment or show watermarks (Shutterstock, iStock, Adobe Stock, Pixel-Shot).
- Photos of people you know, unless they give written consent.

A duplicate check against the training images will run on your Mac before the experiment, as a safety net.

## 4. Fill in `image_sources.csv` (one row per photo)

| Column | Example |
| --- | --- |
| file | `both_head_01_body.jpg` |
| set | `test` or `dev` |
| category | `both_head` |
| true_posture | `head` / `chest` / `abdomen` / `none` (what is **really** in the photo) |
| true_flushing | `yes` / `no` (is the face **really** red?) |
| source_site | `Pexels` |
| page_url | the photo's own page link |
| author | photographer name (if shown) |
| license | `Pexels License` / `CC BY 4.0` / `Public domain` … |
| downloaded_on | `2026-11-02` |
| notes | anything unusual (e.g. "left hand only", "face slightly turned") |

`true_posture` and `true_flushing` are the "correct answers" for the camera part, so look carefully. I will double-check each one.
