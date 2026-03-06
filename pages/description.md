# 🩺 
# **OncoSegment: Breast Tumor Mapping Challenge**

# 💡 
## **Why this Problem is Interesting**

Breast cancer remains one of the leading causes of death among women worldwide. Early and accurate detection through ultrasound is critical, but interpreting these images is notoriously difficult.Ultrasound images suffer from "speckle noise" and low contrast. Much like the "blobs" in plasma physics simulations, tumors in ultrasound are intermittent, have irregular boundaries, and can be easily confused with healthy tissue. This challenge asks you to develop an AI that can precisely outline tumor boundaries, a task that directly assists radiologists in biopsy planning and treatment.

# 📊 
## **The Data**

The dataset is derived from the Breast Ultrasound Images (BUSI) collection, featuring women aged 25–75. It consists of high-quality PNG scans with a standard resolution (approx. 500x500 px).

Dataset Breakdown:

Benign: Tumors that are typically oval or round with smooth edges.

Malignant: Cancerous tumors, often characterized by "spiky" (spiculated) or irregular borders that invade surrounding tissue.

Normal: Scans with no tumors (used to test your model's ability to avoid False Positives).

What you receive in the Starting Kit:
train_images: A list of PIL images (Grayscale) showing the breast 
tissue.
train_masks: Binary masks where 1 represents the tumor and 0 represents the background.
train_labels: Metadata identifying the tumor as benign or malignant.
test_images: Images where you must predict the mask.

 # 📏 
## **Metrics: Measuring Precision**

In medical imaging, a "box" is not enough. We need to know exactly which pixels are cancerous. We use two complementary overlap metrics:
1. Primary Metric: Dice Coefficient (F1-Score)The Dice coefficient is the gold standard for medical segmentation. It is twice the area of overlap divided by the total number of pixels in both the prediction and the ground truth.
$$Dice = \frac{2 \times |P \cap G|}{|P| + |G|}$$
Why? It is highly sensitive to the internal "fullness" of the mask and rewards models that get the shape exactly right.
2. Secondary Metric: Jaccard Index (IoU)
The Jaccard index is the area of intersection divided by the area of union.
$$Jaccard = \frac{|P \cap G|}{|P \cup G|}$$
Why? 
It is more "punishing" than Dice for small errors.
It helps break ties on the leaderboard by highlighting models with the fewest stray False Positive pixels.
