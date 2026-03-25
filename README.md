1. Dataset Description
Starting off, the Wine Quality dataset came from Hugging Face Datasets - I picked that one. It
holds numbers tied to traits such as acidity, leftover sugar, alcohol levels, things like that.
Because it's packed with many numeric layers, clustering methods fit well here. Without using
any given categories, patterns might still show up through grouping techniques. That’s why this
collection works - no need for prior tags.

<img width="809" height="705" alt="image" src="https://github.com/user-attachments/assets/92f9db52-341a-4684-9b8f-ca704130af84" />

3. Groupings and What They Show
● Starting off, data prep involved scaling values so that bigger numbers - say, total sulfur
dioxide - wouldn’t skew how distances were measured. Instead of leaving raw ranges
untouched, each feature got adjusted through StandardScaler. This step balanced
everything before any comparisons took shape down the line.
● Silence settled around the graph when k hit 3 - bent sharply like a knee in snow. The
curve had been falling steadily, pulled down by inertia’s weight until it just stopped
fighting. Three pieces emerged, clean and separate, shaped by distance folded inward.
Not forced apart, but found that way.
● A cloud of dots appeared after squeezing the numbers down using PCA. One glimpse
revealed three main groups floating apart, yet edges blurred where types bled into one
another. Shapes hinted at shifts too fine to draw sharp lines.
● Maybe it's the tart ones that land in Cluster 0 - low on sweetness, sharp on bite. Over in
Cluster 2, though, warmth matters more than zing, leaning into rich, mellow pours.
4. Supervised Model Performance
A different way to check if the groups stay stable? A Random Forest model learned to guess the
group names that K-Means had already given. Instead of just trusting the shapes, the classifier
tested whether patterns held up through prediction. Each label became a clue, not a rule.
Patterns either repeated themselves or they did not. Stability showed up as accuracy across
tries. When guesses matched often, confidence grew. The method leaned on repetition, nothing
more.
