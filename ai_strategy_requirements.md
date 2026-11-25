# AI strategy requirements

Context: many receipt images are labelled, many CSV transactions are
categorised, and some receipts are linked to CSV transactions. That labelled
data is the basis for everything below.

IDs are stable: never renumber, and retire an ID instead of reusing it.

## Cross-inference (CI)

- **CI-1** Every field the user confirms while labelling a receipt is used to
  improve the suggestions for the remaining fields of that receipt. Example:
  once the category is `groceries:ah`, the address suggestion favours the
  known Albert Heijn addresses.
- **CI-2** Before any user input, the AI makes a first guess per field from
  the image alone.
- **CI-3** A suggestion is a ranked list of candidates with a confidence, not
  a single value.
- **CI-4** A suggestion never overwrites a value the user typed or confirmed.
- **CI-5** All cross-inference strategies are kept in one registry (ST-*),
  limited to fields in the current receipt label structure. Each entry has an
  ID, the known fields, the improved field and its tier (TR-1).

## Strategies (ST), seed list

| ID   | Known                  | Improves                              |
| ---- | ---------------------- | ------------------------------------- |
| ST-1 | category               | shop name, address                    |
| ST-2 | shop name              | category                              |
| ST-3 | shop name              | address (known branches)              |
| ST-4 | address                | shop name                             |
| ST-5 | date + total           | linked CSV transaction                |
| ST-6 | linked CSV transaction | shop name, date, total, payment account |
| ST-7 | shop name              | item descriptions, item categories    |
| ST-8 | items                  | category                              |
| ST-9 | items + tax            | subtotal, total                       |

## Evaluation (EV)

- **EV-1** Every model and strategy reports accuracy per field on a held-out
  set it was never trained on.
- **EV-2** The held-out set is the newest receipts (split by time), so the
  result reflects future receipts.
- **EV-3** Every result is compared against a baseline: the most frequent
  value given the known fields. A model that does not beat the baseline is
  not used.
- **EV-4** "Best model" in this document means the highest EV-1 accuracy among
  the models that satisfy HW-3 and the speed limits (SP-*).

## Models and hardware (HW)

- **HW-1** Each model declares its requirements: RAM, VRAM, disk space and
  whether it needs a GPU.
- **HW-2** Models are split into two classes: those that run on any old
  laptop, and those that need more.
- **HW-3** A model "runs flawlessly" on a device when all its declared
  requirements are met with a safety margin and a short benchmark on that
  device meets the speed limit of its use (SP-*).
- **HW-4** By default the code uses the best models (EV-4) that run flawlessly
  on the current device, without asking the user.
- **HW-5** On request, the code lists all models with their requirements next
  to what the device has, with unmet requirements in red. The user can select
  models from that list.
- **HW-6** Available disk space is checked before any model is downloaded or
  loaded. The HW-5 list includes needed and available disk space.
- **HW-7** If the disk cannot hold all selected models, only the best ones
  that fit are kept.
- **HW-8** qvac is added as a model runtime beside the existing Ollama
  pipeline; it does not replace it. Each model declares which runtime it
  uses.

## Training (TR)

- **TR-1** Each strategy is implemented in the cheapest tier that passes EV-3:
  1. priors: lookup and frequency tables built from the labels, rebuilt in
     seconds;
  2. small classifiers: trained in minutes on a CPU;
  3. fine-tuned models: hours to days, optional.
- **TR-2** One command, `--train`, trains everything that is out of date
  (DS-2) on the new labels.
- **TR-3** `--train <names>` limits training to the named strategies (ST-*)
  and/or models.
- **TR-4** Every suggestion the user corrects is logged (suggested value and
  accepted value) and used as training data.

## Dataset tracking (DS)

- **DS-1** A dataset configuration is identified by a hash over its label data
  and the label-schema version.
- **DS-2** Each trained model stores the dataset configuration hash, model
  version and hyperparameters it was trained with. If any of them differs
  from the current ones, the model is out of date.
- **DS-3** Adding or editing a label creates a new dataset configuration;
  older ones stay in the history.
- **DS-4** Dataset configurations are ordered by when they were first
  registered.
- **DS-5** Each dataset configuration is either public or private (contains
  private data). The public dataset is negligible in size right now.

## Training progress view (UI)

- **UI-1** A CLI table shows models as rows and strategies as columns, for one
  dataset configuration at a time.
- **UI-2** Each cell shows five values for that model, strategy and dataset
  configuration, formatted as `23%,352h 45%,21m 87%`:
  1. training completion %;
  2. ETA until training completes;
  3. inference completion %: the share of receipts in that dataset
     configuration the model has processed;
  4. ETA until inference completes;
  5. inference accuracy % on the held-out receipts processed so far (EV-1).

  ETAs are in days, hours or minutes (`34d`, `352h`, `21m`). A finished stage
  shows `100%` without ETA; a value that is not available yet shows `-`.
- **UI-3** The cell background is a gradient that follows the inference
  accuracy.
- **UI-4** Ctrl+Left and Ctrl+Right step through the dataset configurations in
  DS-4 order and wrap around at the ends.
- **UI-5** Private dataset configurations are highlighted orange.
- **UI-6** Tier-1 priors (TR-1) use the same cell; their training shows `100%`
  as soon as they are rebuilt.

## Speed (SP)

- **SP-1** Background extraction of newly added receipt images may take up to
  3 weeks for about 50 receipts. Capable but slow models are fine there.
- **SP-2** Cross-inference during labelling finishes within the time between
  the user entering the known field and reaching the improved field. Example:
  about 23 seconds pass between entering the category and reaching the
  address, because the user reads and enters the amount and account in
  between.
- **SP-3** Tier-1 priors (TR-1) respond within 1 second.
- **SP-4** The labeller records a timestamp every time a field is entered or
  re-edited, per receipt.
- **SP-5** The time budget per strategy (ST-*) is derived from the SP-4
  timestamps. A model may serve a strategy during labelling only if its
  measured inference time fits within that budget.

## Background jobs (BG)

- **BG-1** Background inference and training survive a reboot and resume from
  the last checkpoint without redoing finished work.
- **BG-2** Background work pauses or throttles on battery power and while the
  device is in use.

## Privacy (PR)

- **PR-1** All inference and training run locally. No receipt or transaction
  data leaves the device.
- **PR-2** Training can be limited to the public dataset or include private
  data (binary choice).
- **PR-3** A model trained on any private dataset configuration is tagged
  private and is never exported or published.

## Open questions (OQ)

- **OQ-1** Resolved: qvac runs beside Ollama, see HW-8.
- **OQ-2** Resolved: 23 seconds is intended, see SP-2.
