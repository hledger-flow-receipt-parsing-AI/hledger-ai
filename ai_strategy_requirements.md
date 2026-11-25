

There are quite some labelled receipt images already and there are quite some csv transactions categorised, and there are some receipts linked to csv transactions. 

## cross-inferencing
If a user writes "groceries:ah" as the category of a receipt, then the adres of the receipt is more likely to be an adres that already is in the list of adresses (of the albert heijn shop). So at first, an AI may just try to read the adres from the receipt based on the prior training data, but as soon as, during the labelling the user types the category "groceries:ah" the inferencing AI should have more information to make a more accurate guess.

## cross-inferencing strategies
Keep a list of all possible cross-inferencing and smart insights/labelled information that can be used to enhance the inference of various (sub)components of the receipts. (Within the current receipt label structure).

## Platform
I would like a distinguishment made between AI's that can be ran on any old laptop, and AI's that need more ram etc. Ideally, I want the code to list the of models and then make the requirements red that are not satisfied on the current device (if the user wants to see/select which models is/are used). By default the code should use the best models that run flawlessly on that hardware.

Possibly you could use qvac?

The user should be able to give 1 command "--train" to make the models train on the new labels. Use some kind of hashing system or tracking system to determine on which data each model was trained. Allow the user to specify specific models/inferencing strategies to be trained like --train category-to-address inferencing models etc. (Based on the cross-inferencing strategies list mentioned above, and the AI models mentioned above).

I want to see per model per inference strategy a percentage of how far the training is done per dataset configuration. I want the dataset configurations to be chronological in order of when they were first registered (the hash based tracking system). So show that table in CLI with models rows and strategies on columns. Make ctrl+arrow left right loop through the timeline of the dataset configurations. Show orange highlight which datasets contain private data, allow distinguishing binary between training on private data and the "public dataset" (The public dataset is negilble in size right now).
Besides the 23% indicator in each cell, colour the cell background with a gradient so I can see how far the training per dataset is. Also add the ETA in each cell so 2%,34d or 23%,352h or 45%,21m  for days, hours or minutes.

## Speed
If the user adds receipt images the user may wait up to 3 weeks, it is perfectly fine to have capable but slow models inference 50 receipts over 3 weeks in the background. However for the cross-inferencing mentioned above, it may be necessary to use faster inferencing models because the ai has roughly 23 seconds between e.g. groceries specification and the adres selection.


## Available disk space
Before you load all you can, look at the available disk space, in the requirements also include the needed and available disk space. Take only the best model if more than one model is not possible.

