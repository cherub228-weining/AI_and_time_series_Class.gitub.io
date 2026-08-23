# From VARs to AI-based Time Series Forecasting

Companion site for *From Vector Autoregressions to AI-based Time Series Forecasting: A Review* by Likai Chen and Weining Wang.

The site includes:

- the paper abstract and organizing questions;
- five linked short-course modules;
- an embedded 55-slide lecture deck;
- five prepared class-recording slots;
- six runnable NumPy exercises and figure-generation code;
- a searchable, updateable literature library;
- printable course, exercise, and reference materials.

## Publish with GitHub Pages

This repository is designed for:

`https://github.com/cherub228-weining/AI_and_time_series_Class.gitub.io`

After the files are on the `main` branch, open **Settings → Pages** and choose **GitHub Actions** as the source. The included workflow publishes to:

`https://cherub228-weining.github.io/AI_and_time_series_Class.gitub.io/`

No build command is required.

## Add a new paper

Open [`data/references.json`](data/references.json), copy an existing entry, and change its `id`, `title`, `authors`, `year`, `venue`, `category`, and `url`. Allowed categories are:

- `classical`
- `transformers`
- `foundation`
- `diffusion`
- `evaluation`

Update `lastUpdated` at the top of the file. The search and topic filters update automatically.

## Add class videos

When recording URLs are available, replace the corresponding “Recording link needed” card in `index.html` with a YouTube iframe or an HTML5 `<video>` element. The five slots are in the `#lectures` section.

## Local preview

Serve the folder with any static server, for example:

```bash
python -m http.server 4173
```

Then open `http://127.0.0.1:4173/`.
