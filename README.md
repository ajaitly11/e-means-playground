# k-means vs e-means playground

**Live: https://ajaitly11.github.io/e-means-playground/**

A static web page that runs classic k-means and Meta's e-means side by side in the visitor's browser. No server, no build step, no cost: everything runs on the visitor's device.

Three experiments:

1. **How quickly can each one sort 60,000 handwritten digits?** MNIST at 14×14, live group averages, timers and an error chart.
2. **Does e-means help on a small, simple task like repainting a photo?** 4, 16 or 64 colours. Visitors can load their own photo; it never leaves the device.
3. **Does the time saved grow as the data grows?** The digit task at 1,000, 10,000 and 60,000 digits.

## Files

| File | What it does |
|---|---|
| `index.html` | Page layout |
| `style.css` | Look and motion (Arnitly palette, fonts, easing) |
| `app.js` | Experiments, charts, drawing |
| `engine.js` | k-means (Lloyd) and e-means; shared distance kernel |
| `worker.js` | Runs one algorithm per Web Worker so both race at once |
| `data/` | Digits (`mnist14.bin.gz`, 4 MB) and two CC BY 2.0 photos |
| `tests/check.mjs` | Runs both engines in Node and prints time and error |

## Run it locally

```bash
python3 -m http.server 3232 --directory .
```

Then open http://localhost:3232. Opening `index.html` straight from disk will not work, because browsers block Web Workers on `file://`.

## Put it online for free (GitHub Pages)

1. Create a new public GitHub repository, for example `e-means-playground`.
2. Copy the contents of this folder (including `.nojekyll`) to the repository root and push.
3. In the repository: Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
4. After a minute the page is live at `https://<username>.github.io/e-means-playground/`.

GitHub Pages is free for public repositories, has no server to pay for, and serves these files as they are.

## How faithful is the e-means port?

`engine.js` follows `faiss/contrib/e_means` (faiss 1.16.0): the same batch-size and step-size laws, the cosine step schedule with a 0.1 floor, revival every 5 updates below 0.1 of the mean mass (off at 2 dimensions or fewer), donor splits drawn in proportion to mass, and the last-pass mean at the end. On the 60,000 digits with 100 groups it picks batch 2,048 and step 0.595, the same as Meta's Python code. Final errors land in the same range (Node, 3 seeds: k-means 307,965 to 309,111; e-means 307,272 to 307,818; faiss reference: k-means 307,693 to 308,689; e-means 307,248 to 307,887).

```bash
node tests/check.mjs 60000 100 3
```

## Credits

- Paper: Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501, 2026.
- Digits: MNIST, Yann LeCun, Corinna Cortes and Christopher Burges, CC BY-SA 3.0.
- Photos: "Temple" by danielbuechele and "Flower" by vultilion on Flickr, CC BY 2.0, via scikit-learn.
