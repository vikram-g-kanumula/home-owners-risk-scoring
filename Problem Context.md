<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I want to understand the life cycle of a homeowners insurance policy with specific focus on the risk involved throughout the policy life - I'm interested in understanding the underwriting risk and how the risk is evaluated and the journey to determining the policy premium. Let's focus on understanding what models the gold standard in the US and what are their limitations. I want to understand the opportunities that exist in improving the current version of the standard solution by providing additional intelligence layer over it to capture the residual variance in a glass box approach.

US homeowners pricing today is built around generalized linear models (GLMs) and vendor frameworks like ISO’s by‑peril rating and ISO Risk Analyzer; these are transparent, regulator‑friendly, but leave material unexplained variance that an additional, explainable “intelligence layer” can capture using richer data and modern glass‑box ML/XAI techniques.[^1_1][^1_2][^1_3][^1_4][^1_5]

Below is a conceptual map of the lifecycle, current “gold standard” models in the US, their limitations, and where an interpretable residual‑risk layer can live.

***

## Policy lifecycle and underwriting risk

For a homeowners policy, underwriting risk evolves over several stages.[^1_6][^1_7]

- **New business quote \& bind**: Carrier evaluates the dwelling and applicant (location, construction, age, prior claims, protection, etc.), decides to accept/decline, and sets initial coverage, deductibles, and premium.[^1_8][^1_7][^1_6]
- **Post‑bind inspection \& adjustments**: For selected risks, inspections or external data may reveal condition issues (roof, wiring, plumbing, pools, trampolines, etc.), leading to surcharges, coverage restrictions, or cancellation/non‑renewal.[^1_7][^1_8]
- **In‑force period**: True risk can drift as the insured changes the property (renovations, new pool, renting out rooms) or as environmental risk (crime, wildfire, flood, hurricane) evolves.[^1_8][^1_6]
- **Renewal**: Premium is re‑rated using updated exposure info, loss history, and sometimes refreshed external scores; underwriters may tighten or relax terms based on emerging experience.[^1_6][^1_7]

From the carrier’s perspective, the central underwriting risks are: mis‑pricing (premium vs expected loss + expenses), mis‑classification (wrong segment/territory/peril mix), selection against the insurer, and model risk in the rating plan itself.[^1_9][^1_4]

***

## How risk is evaluated today

Modern homeowners underwriting combines traditional “file‑and‑use” factors with granular third‑party data and predictive models.[^1_2][^1_7][^1_8][^1_6]

Key **risk dimensions** commonly used:

- **Property characteristics**: age, construction type, roof type/condition, square footage, replacement cost, occupancy, updates to plumbing/electrical/HVAC.[^1_7][^1_8][^1_6]
- **Location \& environment**: territory, ISO protection class (fire service), distance to hydrant/fire station, local crime, weather and catastrophe exposure (hail, wind, wildfire, convective storm, etc.).[^1_1][^1_2][^1_8][^1_7]
- **Usage \& hazards**: presence of “attractive nuisances” (pool, trampoline, treehouse), home‑based business, short‑term rental activity.[^1_8][^1_6][^1_7]
- **Behavior \& history**: prior claims (frequency, type, amount), lapses in coverage, sometimes credit‑based insurance scores where allowed.[^1_3][^1_2][^1_8]

Underwriters historically relied on rules plus CLUE reports (loss history) and inspections; now much of this is fed into scoring and rating models that produce expected loss costs by peril or coverage.[^1_2][^1_3][^1_1][^1_8]

***

## Journey from risk to premium

In a typical US carrier, the premium is derived through a multi‑step actuarial/risk‑modeling process.[^1_10][^1_5][^1_3][^1_1][^1_2]

1. **Base loss costs**
    - Actuaries estimate pure premiums (loss cost per unit of exposure) using exposure and historical claims data, often separately for different coverages or perils.[^1_5][^1_11][^1_10]
2. **Risk classification using GLMs (gold standard)**
    - Risk classification has moved from one‑way “relativities” to multivariate **generalized linear models (GLMs)**, which are now widely considered the gold standard for P\&C insurance pricing, including homeowners.[^1_4][^1_5]
    - GLMs are used to model **frequency**, **severity**, or a combined “pure premium” (often with Poisson/negative binomial for frequency and Gamma/Tweedie for cost), with a log link so that rating factors combine multiplicatively.[^1_11][^1_10][^1_5]
3. **By‑peril rating frameworks (ISO as de facto standard)**
    - Historically, homeowners rating often used “all perils combined” loss costs; ISO introduced a **by‑peril rating manual** so separate rates can be developed for wind, fire, theft, liability, etc., improving precision.[^1_1]
    - ISO’s **Risk Analyzer Homeowners** product uses predictive modeling with hundreds of indicators to estimate **non‑catastrophe losses by peril** at policy level and provides “by‑peril” relativities at territory or finer geographic levels.[^1_3][^1_2][^1_1]
4. **Catastrophe modeling overlay**
    - For hurricane, earthquake, wildfire, and other CAT perils, carriers typically rely on external catastrophe models; the CAT component is blended with non‑CAT GLM output when setting total indicated rates, although cats are often handled separately for reinsurance and capital purposes.[^1_9][^1_2][^1_1]
5. **From indicated rate to filed \& written premium**
    - Indicated technical rates are then adjusted for expenses, profit/target return, competitive market position, and regulatory constraints before being filed as a rating plan.[^1_4][^1_5][^1_9]
    - At point of sale, the system applies the filed GLM relativities, by‑peril and coverage relativities, credits/surcharges, and minimum/maximum premium rules to compute the actual premium presented to the customer.[^1_5][^1_2][^1_3][^1_1]

This GLM‑plus‑by‑peril structure, often anchored in ISO manuals and vendor models, is effectively the current “gold standard” for US homeowners risk classification and pricing because it is data‑driven yet explainable and acceptable to regulators.[^1_2][^1_3][^1_4][^1_5][^1_1]

***

## Why GLMs and ISO‑style plans are dominant

Several characteristics make GLMs and ISO‑type rating plans dominant in US homeowners:

- **Regulatory transparency**: GLMs provide explicit, additive log‑linear contributions for each variable, which can be documented in rate filings and explained to regulators, unlike many black‑box ML models.[^1_4][^1_5]
- **Actuarial principle alignment**: They can be calibrated to ensure rates are not excessive, inadequate, or unfairly discriminatory, and they align well with classical actuarial ratemaking frameworks.[^1_5][^1_4]
- **Vendor ecosystem**: ISO’s manuals and Risk Analyzer provide turnkey factors and models, letting carriers “plug in” sophisticated by‑peril, geographic, and environmental factors without building everything in‑house.[^1_3][^1_1][^1_2]

Academic and professional literature explicitly notes that risk classification in P\&C pricing has moved from univariate relativities to GLMs, and that GLMs are viewed as the current industry standard even as ML methods gain interest.[^1_4][^1_5]

***

## Key limitations of current standard models

Even though these models are strong baselines, they leave plenty of residual variance and operational gaps.

### 1. Structural limitations of GLMs

- **Limited interaction structure**: GLMs typically include a modest number of manually selected interactions; many real‑world patterns (e.g., interaction of roof age, local tree cover, and specific storm paths) are higher‑order and nonlinear, which GLMs struggle to capture without becoming unwieldy.[^1_10][^1_3][^1_5]
- **Assumed functional form**: Log‑linear effects and simple splines mean that complex response curves (e.g., non‑monotonic risk vs. home age) may be poorly approximated or require hand‑crafted transformations.[^1_10][^1_5]
- **Data‑sparse segments**: Certain peril/coverage combinations, endorsements, or rare risk types have sparse data; GLMs either over‑smooth (hiding true heterogeneity) or become unstable in those segments.[^1_11][^1_10]


### 2. Separation of CAT and non‑CAT

- Non‑CAT frequency/severity GLMs and external catastrophe models are often calibrated separately, so correlations between “attritional” and CAT risk at property level (e.g., roof quality affecting both hail frequency and hurricane losses) may be under‑captured.[^1_9][^1_1][^1_2]


### 3. Data and feature limitations

- **Coarse geography and territory structures**: Rating territories and even ZIP‑code‑level pricing miss micro‑level variation that address‑level environmental data can capture; ISO Risk Analyzer begins to address this, but most carriers still use a relatively coarse filed structure.[^1_1][^1_2][^1_3]
- **Static snapshots**: Many rating plans treat variables as static (e.g., roof age bucket) and are slow to incorporate real‑time signals such as recent building permits, roof replacements, vegetation management, or IoT‑detected water leaks.[^1_12][^1_2]


### 4. Adoption barriers for more powerful ML

- More flexible ML models (gradient boosting, deep nets) can outperform GLMs in prediction, but adoption in pricing is limited by concerns around transparency, regulatory acceptance, governance, and bias control.[^1_13][^1_12][^1_4]
- Industry analyses explicitly note that lack of interpretability has slowed use of ML in ratemaking even as such models gain traction in reserving and other back‑office tasks.[^1_4]

These constraints create exactly the “residual variance” and friction points that an additional intelligence layer could target.

***

## What a “glass‑box” intelligence layer could look like

You can think of the standard GLM/ISO framework as a strong baseline model, and the new intelligence layer as a **constrained, explainable residual model plus decision‑support tooling** that sits on top.

### Targeted objectives for the extra layer

- Explain and capture **residual risk** not explained by the filed GLM/ISO factors.
- Provide **interpretable uplift/discount signals** for pricing, underwriting decisions, and inspection prioritization.
- Enhance **per‑peril and per‑coverage understanding** (e.g., non‑weather water, liability, wildfire) where current models are weakest.[^1_11][^1_10][^1_2][^1_1]
- Maintain **regulatory‑grade transparency**, using techniques from explainable AI (XAI) research in insurance underwriting and pricing.[^1_12][^1_13][^1_4]

***

## Modeling approaches for the residual layer (glass‑box)

Drawing on recent work in explainable AI for underwriting and pricing, several architectures are promising.[^1_13][^1_12][^1_4]

### 1. Interpretable ML on top of GLMs

- **Generalized additive models (GAMs) / GAMs with pairwise interactions (GA2Ms)**
    - These extend GLMs by learning flexible, smooth, univariate (and sometimes bivariate) functions for each feature, while still summing contributions in a way that is visually interpretable.[^1_4]
    - The residual layer could be a GAM trained on the **residuals of the filed GLM**, constrained to be small (e.g., uplift factor around 1) and monotone where required, so each feature’s marginal effect can be plotted and filed if needed.[^1_5][^1_4]
- **Monotone gradient‑boosted trees with XAI**
    - Tree‑based ensembles (e.g., gradient boosting) can be constrained to satisfy monotonicity and other business rules, and then explained using SHAP, accumulated local effects, and surrogate rule sets.[^1_12][^1_13][^1_4]
    - XAI frameworks in underwriting show that combining SHAP, rule extraction, counterfactuals, and natural‑language explanations can make such models acceptable to regulators and business users while significantly improving accuracy.[^1_12]


### 2. Multi‑layer XAI architecture

Recent underwriting‑automation research proposes **multi‑layer XAI frameworks** that combine several explanation methods (SHAP, ALE, rule extraction, counterfactual explanations) to maintain transparency even with sophisticated models.[^1_12]

A similar architecture for homeowners pricing could consist of:

- **Risk‑prediction layer**: A constrained ML model predicts residual loss cost or risk score given all baseline rating variables plus new granular features (property imagery, environmental indices, IoT signals).
- **Explanation layer**:
    - Global: variable importance, shape functions, interaction heatmaps.
    - Local: SHAP value breakdowns, rules (“If roof age > 20 years and tree‑overhang index high, uplift +15% for wind/hail hazard”), and counterfactuals (“If roof replaced, predicted non‑CAT loss cost down X%”).[^1_13][^1_12][^1_4]
- **Narrative layer**: Automatically generated plain‑language rationales that underwriters and regulators can read in filings or exception reviews.[^1_13][^1_12]

***

## Data opportunities for capturing residual variance

The intelligence layer is only as good as the additional signal it consumes; several data domains are natural candidates.

### 1. High‑resolution environmental \& property data

- Vendor models already show gains from address‑level environmental variables (e.g., terrain, local weather history, wildfire fuel, hail swaths) compared to territory‑level pricing.[^1_2][^1_3][^1_1]
- An intelligence layer could ingest:
    - Satellite and aerial imagery derived features (roof condition, material, slope, surrounding vegetation).
    - Building footprints, elevation, parcel‑level flood indices, proximity to brush or industrial hazards.
    - Fine‑scale crime and fire‑risk indices, beyond coarse ISO protection classes.[^1_3][^1_1][^1_2]


### 2. Temporal \& behavioral signals

- **Change detection**: Building permits, property‑listing data, or image‑based change detection can identify renovations, new pools, or roofing work that materially change risk in the middle of a policy term, where current rating plans usually lag.[^1_2][^1_12]
- **IoT and smart‑home data**: Leak detectors, monitored alarms, and other telematics‑style signals can materially affect non‑weather water and theft risk, but are often captured today only as crude discounts rather than fully modeled risk factors.[^1_13][^1_12]

***

## Concrete “glass‑box” use cases

Here are a few specific ways to add an intelligence layer without breaking transparency.

### 1. Residual uplift factor by peril

- Train a constrained residual model that outputs a **multiplicative uplift factor by peril** relative to the GLM + ISO base.
- File the structure as a function of clearly documented features (e.g., imagery‑derived roof condition index, wildfire defensible‑space score), with GAM or monotone‑GBM plots as exhibit.[^1_1][^1_3][^1_2][^1_4]
- Use XAI to show, at both global and local levels, why a given risk attracts a modest uplift or credit versus the baseline GLM.[^1_12][^1_13][^1_4]


### 2. Inspection prioritization layer

- Use an ML model (explainable via SHAP/rules) to rank policies by **information gain from inspection**—i.e., where a physical inspection is most likely to change premium or acceptability.[^1_12]
- This model does not directly set price but helps allocate inspections, so regulators are typically more comfortable with more flexible methods, as long as decisions are explainable and non‑discriminatory.[^1_13][^1_12]


### 3. Coverage and deductible personalization

- XAI‑enhanced underwriting research shows that explainable models can power **personalized policy recommendations** (coverage mix, deductibles, exclusions) while producing clear rationales for each suggestion.[^1_12]
- For homeowners, a similar engine could propose higher water‑damage deductibles for older‑plumbing homes, or recommend specific endorsements (e.g., service‑line coverage) where exposure indicators are high—always paired with human‑readable explanations.[^1_10][^1_11][^1_12]

***

## Governance and filing considerations

Because you are proposing an overlay to a regulated pricing system, governance is as important as modeling.

- **Model risk management**: Treat the residual model as a separate component with its own validation, monitoring, and bias testing, as emphasized in ML‑in‑pricing research.[^1_4]
- **Regulatory narrative**: Leverage XAI techniques to produce documentation and examples that show how the model adheres to actuarial standards and avoids prohibited rating factors; recent work demonstrates regulators are more receptive when explanations are systematic and measurable (e.g., using surrogate‑model alignment or explanation‑fidelity metrics).[^1_4][^1_12]
- **Bounded influence**: Implement hard constraints so the intelligence layer can only adjust premiums within a limited corridor around the GLM indication, especially in early deployment; this eases regulatory and internal‑stakeholder concerns.[^1_5][^1_4]

***

If you tell me roughly what role you’re in (e.g., actuarial, data science, product) and whether you want to focus more on model design or regulatory/filing strategy, I can sketch a more concrete target architecture for such a glass‑box residual‑risk system.
<span style="display:none">[^1_14][^1_15]</span>

<div align="center">⁂</div>

[^1_1]: https://www.verisk.com/company/newsroom/iso-introduces-new-homeowners-by-peril-rating-manual-for-better-precision-in-pricing-homeowners-risks/

[^1_2]: https://www.verisk.com/insurance/products/iso-risk-analyzer-suite/iso-risk-analyzer-homeowners/

[^1_3]: https://www.casact.org/sites/default/files/presentation/rpm_2010_handouts_pl3-cummings.pdf

[^1_4]: https://ar5iv.labs.arxiv.org/html/2003.10674

[^1_5]: https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf

[^1_6]: https://openly.com/the-open-door/articles/home-insurance-underwriting

[^1_7]: https://davies-group.com/northamerica/knowledge/how-insurance-risk-factors-into-insurance-underwriting/

[^1_8]: https://www.irmi.com/articles/expert-commentary/the-underwriting-submission-homeowners-insurance

[^1_9]: https://www.uschamber.com/assets/documents/20253125-CCMC-Insurance-White-Paper-Final-digital.pdf

[^1_10]: https://variancejournal.org/article/115777

[^1_11]: https://variancejournal.org/article/115777-rating-endorsements-using-generalized-linear-models

[^1_12]: https://eajournals.org/wp-content/uploads/sites/21/2025/05/Explainable-AI-Enhanced-Underwriting.pdf

[^1_13]: https://www.meegle.com/en_us/topics/explainable-ai/explainable-ai-for-insurance

[^1_14]: https://www.linkedin.com/pulse/explainable-ai-pc-insurance-turning-predictive-power-johnson-mba-i6dsc

[^1_15]: https://eajournals.org/ejcsit/wp-content/uploads/sites/21/2025/05/Explainable-AI-Enhanced-Underwriting.pdf


---

# I'm a Data Scientist looking for opportunity in improving the variance the current baseline GLM capture. I think the Interpretable ML on top of GLMs sounds promising to me specifically the "Generalized additive models (GAMs) / GAMs with pairwise interactions (GA2Ms)". Does this satisfy the requirement in terms of explaining exactly how the additional variance being captured in mapped to the interactions or features over and above the current baseline GLM model? If yes, can we deep dive on this?

Yes – if you design it correctly, a GAM/GA2M layer on top of your existing GLM can both capture additional variance and let you point *exactly* to which feature or interaction is responsible for the uplift, in a glass‑box way. GAMs and GA2Ms are explicitly built so the prediction is decomposed into additive main‑effect and pairwise‑interaction functions that you can visualize and attribute one‑for‑one to features. In insurance pricing, vendors and practitioners already use GAM‑structured models for transparent, regulator‑friendly rating, precisely because their effects can be decomposed and inspected term by term.[^2_1][^2_2][^2_3][^2_4][^2_5][^2_6][^2_7][^2_8]

Below is a focused deep dive tailored to your “residual on top of GLM” use case.

***

## What GAM / GA2M gives you, structurally

A standard GLM for, say, log pure premium looks like:
$\eta_{\text{GLM}}(x) = \beta_0 + \sum_j \beta_j x_j$ (possibly with a few hand‑crafted interactions).

A **GAM** replaces linear terms $\beta_j x_j$ with learned univariate shape functions $f_j(x_j)$:

$$
g(\mathbb{E}[Y \mid x]) = \beta_0 + \sum_j f_j(x_j)
$$

where $g$ is the link (log for pure premium, etc.).[^2_4][^2_7][^2_9]

A **GA2M** (Caruana et al., “Accurate intelligible models with pairwise interactions”) further adds a small set of pairwise interaction functions:[^2_2][^2_5][^2_4]

$$
g(\mathbb{E}[Y \mid x]) = \beta_0 + \sum_j f_j(x_j) + \sum_{(i, k) \in S} f_{ik}(x_i, x_k)
$$

- Each $f_j$ is a 1‑D function you can plot; each $f_{ik}$ is a 2‑D surface/heatmap.
- Frameworks like Explainable Boosting Machines (EBMs) implement GA2Ms via boosted shallow trees, but still enforce this additive‑plus‑pairwise structure so contributions remain exactly decomposable.[^2_5][^2_10][^2_4]

This structure is why these models are classed as “inherently interpretable” or “glass‑box”.[^2_11][^2_12][^2_4][^2_5]

***

## How to put GAM / GA2M *on top* of a baseline GLM

To satisfy “over and above the current baseline GLM”, you want the GLM to remain the baseline and the GAM/GA2M to *only* explain residual structure. There are two main designs.

### Option A (strongest for your requirement): residual GAM with GLM as fixed offset

1. **Freeze your filed GLM**
    - Keep its form and parameters as‑is; treat its linear predictor $\eta_{\text{GLM}}(x)$ as a fixed offset.[^2_13][^2_1]
2. **Define a residual target**
    - Common choices:
        - Additive residual on the link scale:
$\delta(x) = g(\mathbb{E}[Y \mid x]) - \eta_{\text{GLM}}(x)$.
        - Or a multiplicative residual factor:
$R(x) = \mathbb{E}[Y \mid x] / \hat{Y}_{\text{GLM}}(x)$, then work on $\log R(x)$.
    - In practice you fit a GAM/GA2M to $\delta(x)$ or $\log R(x)$ with the GLM linear predictor included as an **offset term** in the model.[^2_7][^2_9][^2_4]
3. **Fit GAM/GA2M to residuals**
    - Model:

$$
\delta(x) \approx \sum_j f_j(x_j) + \sum_{(i,k)\in S} f_{ik}(x_i, x_k)
$$

where you *exclude* any terms already perfectly captured by the GLM, focusing on features or transformations you believe are under‑modeled (e.g., geospatial smooths, non‑linear age effects, higher‑order interactions, imagery scores).[^2_3][^2_9][^2_4]
    - Regularize heavily and limit $|S|$ to a small number of interactions to keep the residual model parsimonious and explainable.[^2_2][^2_4][^2_5]
4. **Final predictor**
    - On the link scale:

$$
\eta_{\text{final}}(x) = \eta_{\text{GLM}}(x) + \delta_{\text{GAM}}(x)
$$
    - Or equivalently in multiplicative form on the original scale:
$\hat{Y}_{\text{final}}(x) = \hat{Y}_{\text{GLM}}(x) \times \exp(\delta_{\text{GAM}}(x))$.

Here, **by construction**, *every bit of additional variance explained beyond the GLM* lives inside the sum of $f_j$ and $f_{ik}$ functions, so you can attribute uplift cleanly to those terms.

### Option B: full GA2M replacement, GLM used only for comparison

You refit a full GA2M to the original target instead of residuals:[^2_2][^2_4][^2_5][^2_7]

$$
g(\mathbb{E}[Y \mid x]) = \beta_0 + \sum_j f_j(x_j) + \sum_{(i,k)\in S} f_{ik}(x_i, x_k)
$$

Then:

- Use the old GLM as a **benchmark** rather than a component.
- You still get exact attributions within the GA2M, but you lose the strict “this portion is *over and above* the GLM” semantics: some variance the GLM used to explain might be re‑distributed into non‑linear $f_j$ or interactions.

For ratemaking where the filed GLM is a regulatory anchor, Option A (offset + residual GAM) aligns better with your stated requirement.

***

## Does this meet the “exact mapping” requirement?

Under the residual‑GAM design (Option A), *yes, as close as you can reasonably get*:

- **Global level**
    - Each univariate $f_j(x_j)$ is a curve you can plot: “As roof‑age‑index increases from 0–1, the residual log‑loss factor moves from −0.05 to +0.12.”
    - Each interaction $f_{ik}(x_i, x_k)$ is a 2‑D heatmap: “The combination of high wildfire‑fuel index and long eave overhang adds a +0.20 residual log‑loss.”
    - The sum of *all* such functions equals the modeled residual on the link scale, so you have a complete decomposition of the extra explained variance.[^2_4][^2_5][^2_7][^2_2]
- **Local (per‑policy) level**
    - For any risk $x$, the final prediction can be written as:
$\eta_{\text{final}}(x) = \eta_{\text{GLM}}(x) + \sum_j f_j(x_j) + \sum_{(i,k)\in S} f_{ik}(x_i, x_k)$.
    - You can show an underwriter: baseline GLM indication, plus a table of additive adjustments from each univariate and interaction term, summing to the final uplift.[^2_10][^2_5][^2_4]
    - EBMs and similar GA2M implementations are explicitly marketed as offering exact per‑feature contribution breakdowns for every prediction.[^2_5][^2_10][^2_4]
- **Regulatory/filing perspective**
    - Industry commentary stresses that GAM‑structured models can be “decomposed and directly analyzed piece by piece” and are therefore aligned with fairness, accountability, compliance, transparency, and security (“FACTS”) principles regulators expect.[^2_6][^2_1][^2_11]
    - Because your residual model retains GLM/GAM structure and a standard link and distribution, it fits naturally into existing actuarial documentation frameworks.[^2_8][^2_1][^2_3][^2_6]

There are still two caveats you’ll want to manage (see “Design \& caveats” section below), but conceptually, GA2M residuals give you the mapping you’re looking for.

***

## How this actually improves variance capture (intuitively)

GLMs struggle with nonlinearities and higher‑order interactions unless you manually engineer them; GAM/GA2M lets you learn them automatically but in a constrained way.

Evidence from both generic and insurance‑specific studies shows:[^2_9][^2_1][^2_3][^2_8][^2_4]

- **Nonlinear main effects** – e.g., risk vs. building age often has a U‑shape; GAMs model that curve directly instead of forcing a linear or simple stepwise approximation.
- **Localized geospatial effects** – GAM frameworks have been used specifically to better incorporate geospatial data into pricing while remaining interpretable, improving territory segmentation vs. standard GLMs.[^2_3]
- **Targeted pairwise interactions** – GA2Ms add only a small set of pairwise interactions, chosen by importance, which often capture key effects like “roof material × hail exposure” or “plumbing age × freezing climate” without exploding model complexity.[^2_2][^2_4][^2_5]

Empirical work (e.g., GA2M, EBM, GAMI‑Net) finds that additive‑with‑interactions models can approach or match black‑box methods in predictive accuracy while staying interpretable.[^2_14][^2_4][^2_5][^2_2]

***

## Practical design choices and caveats

To make this work well in a homeowners pricing context, here are key design points.

### 1. Identifiability and “what counts as residual”

- If you *refit* the GLM and GAM jointly, the split between “baseline” and “additional” variance is not unique.
- Using the filed GLM as a **fixed offset** makes the separation well‑defined: all additional structure the model learns beyond the GLM is in the GAM terms.

You’ll still want to enforce centering constraints (e.g., each $f_j$ sums to zero over training data) so the intercept and residual terms are identifiable and consistent.[^2_7][^2_9][^2_4]

### 2. Controlling complexity

- Limit the number of pairwise interactions and apply heredity constraints (only allow an interaction if at least one parent main effect is significant), as done in GAMI‑Net and GA2M‑style models.[^2_14][^2_5][^2_2]
- Use smoothness penalties or shallow tree ensembles so functions do not become too wiggly; this improves both generalization and interpretability.[^2_9][^2_4][^2_5]


### 3. Feature selection and domain constraints

- Prioritize features where you *expect* the GLM to be weak: continuous geography, environmental indices, imagery/IoT features, high‑cardinality variables, etc.[^2_1][^2_3][^2_9]
- Impose domain constraints (e.g., monotone increasing risk with wildfire‑fuel index, non‑negativity for certain uplift terms) where needed for regulatory comfort and business sense.[^2_11][^2_6][^2_1]


### 4. Integration into pricing and governance

- Start by using the residual GAM as a **non‑filed analytic tool** to study residual structure and propose manual GLM improvements; this builds trust.
- Once stable, file the GAM/GA2M layer as a rating factor or set of factors, with:
    - Global plots for each main effect and interaction.
    - Examples of per‑policy decompositions.
    - Sensitivity and stability checks (e.g., how much uplift varies for near‑identical risks).[^2_6][^2_8][^2_1][^2_11]

***

## If you want to move forward from here

If you share a bit about your current stack (R/mgcv, Python, SAS, proprietary tools) and whether you can treat the GLM as a fixed offset in production, I can sketch a very concrete workflow:

- exact training target definition (frequency, severity, pure premium residual),
- library choices (e.g., EBM vs classical spline GAM vs GAMI‑Net‑style NN),
- model diagnostics to prove variance gain and interpretability,
- and how to turn GA2M components into filing‑ready exhibits.
<span style="display:none">[^2_15][^2_16]</span>

<div align="center">⁂</div>

[^2_1]: https://www.celent.com/en/directory/companies/akur8/insights/429021251

[^2_2]: https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf

[^2_3]: https://support.sas.com/resources/papers/proceedings16/8441-2016.pdf

[^2_4]: http://www.mathematology.com/courses/DataMining/Slides/EBM.pdf

[^2_5]: https://www.emergentmind.com/topics/explainable-boosting-machines

[^2_6]: https://www.celent.com/insights/429021251

[^2_7]: https://christophm.github.io/interpretable-ml-book/extend-lm.html

[^2_8]: https://kurser.math.su.se/pluginfile.php/20130/mod_folder/content/0/Kandidat/2020/2020_5_report.pdf?forcedownload=1

[^2_9]: https://aktuar.de/en/knowledge/specialist-information/detail/claim-frequency-modeling-in-insurance-pricing-using-glm-deep-learning-and-gradient-boosting/

[^2_10]: https://people.orie.cornell.edu/mru8/orie4741/lectures/Tutorial4MadeleineUdellClass_2020Dec08_RichCaruana_IntelligibleMLInterpretML_EBMs_75mins.pdf

[^2_11]: https://support.sas.com/resources/papers/proceedings20/4502-2020.pdf

[^2_12]: https://arxiv.org/pdf/2504.10169.pdf

[^2_13]: https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf

[^2_14]: https://ar5iv.labs.arxiv.org/html/2003.07132

[^2_15]: https://www.youtube.com/watch?v=u-2MaBH5NDs

[^2_16]: https://arxiv.org/pdf/2106.02836v2.pdf


---

# Can we use light GBM as the additional intelligence layer to account for the residual variance? If yes, how?

Yes, you can use LightGBM as the additional intelligence layer on top of your GLM to explain and capture residual variance; the main work is in (1) how you define the target (residual vs full), (2) how you control the model’s behavior (monotonicity, regularization), and (3) how you wrap it with XAI so it remains glass‑box enough for pricing. Gradient boosting (including LightGBM) has already been used in P\&C pricing to detect GLM mispricing and to model residuals, with SHAP‑based explanations providing global and local interpretability.[^3_1][^3_2][^3_3][^3_4]

Below is a concise, DS‑oriented blueprint.

***

## 1. Concept: LightGBM as residual learner

Design it analogous to the residual GAM idea:

- Keep your **filed GLM** as the baseline, not replaced.
- Define a **residual target**:
    - On log scale: $\delta(x) = \log Y - \eta_{\text{GLM}}(x)$, or
    - A multiplicative factor $R(x) = Y / \hat{Y}_{\text{GLM}}(x)$ and work on $\log R(x)$.
- Train **LightGBM** to predict this residual, using all the features you believe the GLM under‑uses (geospatial, imagery scores, engineered interactions, etc.).
- Final prediction:
    - $\hat{Y}_{\text{final}}(x) = \hat{Y}_{\text{GLM}}(x) \times \exp(\hat{\delta}_{\text{LGBM}}(x))$.

This is exactly what some insurance research suggests: use tree‑based boosting to find non‑randomness in GLM residuals and identify missing interactions or functional forms.[^3_2][^3_5][^3_4]

***

## 2. Training recipe

Minimal but robust recipe:

1. **Compute GLM predictions and residuals**
    - From your GLM (frequency, severity, or pure premium), compute predicted mean $\hat{\mu}_{\text{GLM}}$ on train/val.
    - Define residual target (e.g., log‑ratio).
2. **Prepare LightGBM inputs**
    - Start with same features as GLM plus any new granular ones (e.g., address‑level peril scores).
    - Consider excluding variables tightly constrained in rating (e.g., territory factor) if you do not want the tree model to “fight” them; or include but later check for conflicts via monotonic constraints / SHAP.
3. **Fit LightGBM on residuals**
    - Task type: regression on $\delta(x)$ (or Tweedie/Poisson objective if you want to be closer to GLM structure).
    - Regularization: small `num_leaves`, relatively shallow trees, conservative learning rate, early stopping.
    - Optionally use **monotone_constraints** for features that must have directional effects (e.g., more hazard → higher residual risk).[^3_6][^3_7][^3_1]
4. **Evaluate**
    - Check improvement in deviance / log‑lik / Gini / RMSE vs GLM alone.
    - Plot residuals before vs after LGBM layer to ensure remaining structure is small and mostly noise.

***

## 3. Making LightGBM “glass‑box enough”

Pure LightGBM is not inherently as interpretable as GA2M/EBM, but you can get quite far with constraints and XAI:

### Model‑side constraints

- **Monotonicity**: LightGBM supports per‑feature monotone constraints, ensuring predicted residuals move in expected directions; violations at training time are pruned.[^3_1][^3_6]
- **Simplicity**: cap tree depth and `num_leaves`, and limit number of boosting iterations to keep the ensemble relatively small and smoother.[^3_7]
- **Feature list**: restrict high‑risk or legally sensitive features; keep to variables already acceptable in GLM pricing plus a few clear new ones.


### Explanation layer

- Use **SHAP** (TreeExplainer) on the LightGBM model: this is a standard approach in recent actuarial/insurance studies to explain gradient‑boosted pricing models globally and locally.[^3_3][^3_4][^3_2]
    - Global: feature importance, SHAP dependence plots → “for wildfire_score, residual factor grows roughly exponentially beyond threshold T.”
    - Local: per‑policy SHAP summaries → contribution of each feature to $\hat{\delta}_{\text{LGBM}}(x)$, which you can present as additive adjustments on top of GLM.
- Aggregate SHAP across segments to propose **back‑porting** stable patterns into an updated GLM or GAM later (e.g., adding an interaction or spline). This is exactly the workflow suggested in “Using Gradient Boosting to Identify Pricing Errors in GLM‑Based …”.[^3_2]

In short, LightGBM + SHAP can be made explainable enough for internal pricing analytics and even, with care, for filed rating justifications.

***

## 4. Two deployment patterns

### Pattern A: analytic assistant, GLM only in production (low risk)

- Use LightGBM purely offline to explore residual structure:
    - Train on GLM residuals.
    - Use SHAP to identify systematic under/over‑pricing regions and missing interactions.
    - Translate stable findings into **new GLM (or GAM) terms** and re‑file the tariff.
- This is what some P\&C research recommends as a practical compromise: leverage GBMs for discovery, not direct pricing.[^3_4][^3_2]

Pros: regulator‑friendly, clear story.
Cons: you’re not using the full predictive power of LightGBM in real time.

### Pattern B: bounded uplift factor in production (higher gain, more governance)

- Treat $\exp(\hat{\delta}_{\text{LGBM}}(x))$ as an **uplift factor** applied to the GLM indication.
- Impose **bounds**, e.g. uplift between 0.85 and 1.15 in early phases.
- For filing / governance:
    - Document LightGBM spec, residual target, validation metrics.
    - Provide SHAP‑based global plots and local example decompositions to show consistency and lack of unfair discrimination.
    - Show that uplift distribution is tight and stable across time and segments.

Pros: leverages more of LGBM’s lift; still anchored in GLM.
Cons: more regulatory scrutiny; must manage stability and fairness carefully.

***

## 5. When would you still prefer GA2M/EBM over LightGBM?

- If your **primary** requirement is “every effect should be plottable as a 1‑D or 2‑D function”, GA2M/EBM is a better fit out of the box.[^3_8][^3_9]
- LightGBM + SHAP is *post‑hoc* interpretable: you can explain, but the model isn’t structurally additive, so explanations are approximations, not the literal model form.
- A hybrid approach some teams use:
    - Start with LightGBM on residuals to explore structure.
    - Then distill learned patterns into a GA2M/EBM or GAM that becomes the filed residual model.

***

If you tell me which of the two patterns (offline assistant vs bounded uplift) you’re leaning toward and your current tech stack (Python/R/others), I can outline a concrete, step‑by‑step implementation plan (features, objectives, metrics, and key checks) for integrating LightGBM as your residual intelligence layer.
<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_16][^3_17]</span>

<div align="center">⁂</div>

[^3_1]: http://ethen8181.github.io/machine-learning/trees/monotonic.html

[^3_2]: https://www.diva-portal.org/smash/get/diva2:1669166/FULLTEXT01.pdf

[^3_3]: https://aktuar.de/en/knowledge/specialist-information/detail/claim-frequency-modeling-in-insurance-pricing-using-glm-deep-learning-and-gradient-boosting/

[^3_4]: https://eforum.casact.org/article/73245-machine-learning-and-ratemaking-assessing-performance-of-four-popular-algorithms-for-modeling-auto-insurance-pure-premium

[^3_5]: https://www.casact.org/sites/default/files/presentation/rpm_2013_handouts_paper_1645_handout_874_0.pdf

[^3_6]: https://www.youtube.com/watch?v=EgKAGrvF8rs

[^3_7]: https://lightgbm.readthedocs.io/en/latest/Parameters.html

[^3_8]: http://www.mathematology.com/courses/DataMining/Slides/EBM.pdf

[^3_9]: https://www.emergentmind.com/topics/explainable-boosting-machines

[^3_10]: http://ui.adsabs.harvard.edu/abs/2022SPIE12330E..0DZ/abstract

[^3_11]: https://github.com/microsoft/LightGBM/issues/3905

[^3_12]: https://www.informatica.si/index.php/informatica/article/download/8100/4457/0

[^3_13]: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html

[^3_14]: https://ui.adsabs.harvard.edu/abs/2022SPIE12330E..0DZ/abstract

[^3_15]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12330/123300D/Research-on-auto-insurance-pricing-based-on-LightGBM/10.1117/12.2646305.pdf

[^3_16]: https://www.celent.com/en/directory/companies/akur8/insights/429021251

[^3_17]: https://www.forbes.com/councils/forbestechcouncil/2024/10/28/harnessing-ai-and-gradient-boosting-for-insurance-premium-modeling/


---

# Let's say we have ~20 features that are used to model baseline GLM out of our universe of 35 features. How do we go about selecting the exact input features to be used in Light GBM?

Think of it in two passes: (1) respect what your GLM already uses, and (2) let the tree model tell you, with guardrails, which of the remaining (and overlapping) features actually help on the *residual* problem.

A practical way to do that for your 35‑feature universe (20 in GLM, 15 not) is:

***

## 1. Decide the *role* of the 20 GLM features

Before selection, be clear on how you want the residual model to relate to the GLM:

- **Conservative option**:
    - Treat the GLM as “owning” those 20 features.
    - Let LightGBM see *all 35* features, but when you interpret and/or back‑port findings, focus on:
        - new features (the 15 not in GLM), and
        - interactions / non‑linearities the GLM did not capture.
    - This is similar to how gradient boosting is used to identify pricing errors and missing interactions in GLM‑based models.[^4_1]
- **Stricter offset option**:
    - If you really want the LGBM to only capture *orthogonal* information, you can:
        - Include the GLM **prediction or linear predictor** as a single feature/offset,
        - Down‑weight or even exclude some of the 20 GLM features from LGBM if they just duplicate that information.

In practice, most actuarial/ML work that boosts GLMs lets the tree model see the same predictors plus extras, then uses its structure/SHAP to highlight where GLM is weak.[^4_2][^4_3][^4_1]

***

## 2. Baseline: let LightGBM do embedded feature “selection”

Tree‑based boosting has **embedded** feature selection: during training, features that don’t help reduce loss simply don’t get used in splits.[^4_4][^4_2]

For your first cut:

1. Train a **regularized LightGBM** on the residual target with *all 35 features*.
2. Inspect:
    - `feature_importances_` (gain‑based, split‑based)
    - SHAP global importance and dependence plots.

Empirical work in insurance pricing shows gradient boosting can surface which predictors and interactions matter most, then those can be added back into a GLM or used to refine the model.[^4_3][^4_5][^4_1][^4_2]

This gives you an initial ranking without prematurely throwing features away.

***

## 3. Then prune with simple, explicit rules

Once you have importance rankings:

- **Drop clearly useless signals**
    - Features never or almost never used in splits, with near‑zero importance, can safely be removed; tree‑based feature selection is a standard pattern.[^4_6][^4_7][^4_4]
- **Watch redundancy with GLM terms**
    - If a GLM feature and a closely related engineered feature both show importance, prefer the more interpretable/controllable one for the residual model (often the engineered or binned version), and keep the GLM version only in the GLM.
- **Limit risky features**
    - Exclude features that are problematic from a governance or fairness perspective, even if they look predictive.

A typical target would be to end up with, say, 15–25 LightGBM inputs that:

- include all 15 “new” features that show signal, plus
- a handful of GLM features that exhibit strong interactions / nonlinearities in SHAP plots.

***

## 4. Use gradient boosting *itself* as a feature‑discovery tool

There is a nice loop here, used explicitly in insurance research: fit boosting, inspect, then update GLM.[^4_1][^4_3]

Concretely:

1. Fit LightGBM on residuals with all 35 features.
2. From SHAP \& interaction plots, identify:
    - Variables with strong marginal effects the GLM currently treats linearly or coarsely.
    - Pairs with strong interactions.
3. Decide for each:
    - “Do we let LGBM keep this as part of the residual layer?”
    - “Or do we back‑port it into a new GLM/GAM term and simplify the residual model?”

This way, LightGBM’s feature usage guides you rather than you hand‑picking the 15 extra features up front.

***

## 5. Guardrails specific to the residual‑layer use case

Given your context (homeowners pricing, GLM as baseline):

- **Always include**:
    - Any new, high‑value signals (e.g., fine‑grained peril scores, imagery‑based indicators) – that’s the whole point of the layer.
- **Usually include** (at least initially):
    - The 20 GLM features as candidates, but expect LGBM to down‑weight those whose effect is already well modeled.
- **Constrain via hyperparameters and monotonicity**:
    - Keep depth / leaves modest and set monotone constraints where you have clear priors; that acts as a soft form of feature discipline and keeps the model behavior sane.[^4_8][^4_9][^4_10]

If you tell me what kinds of 15 extra features you have (e.g., address‑level hazards, behavioral, IoT, credit proxies), I can suggest a more concrete selection strategy and which ones I’d force‑include versus let LGBM decide.
<span style="display:none">[^4_11][^4_12][^4_13][^4_14][^4_15][^4_16][^4_17][^4_18][^4_19]</span>

<div align="center">⁂</div>

[^4_1]: https://www.diva-portal.org/smash/get/diva2:1669166/FULLTEXT01.pdf

[^4_2]: https://arxiv.org/pdf/1508.06378.pdf

[^4_3]: https://www.math.mcgill.ca/yyang/demo.html

[^4_4]: https://apxml.com/courses/intro-feature-engineering/chapter-6-feature-selection/embedded-tree-importance

[^4_5]: https://eforum.casact.org/article/73245-machine-learning-and-ratemaking-assessing-performance-of-four-popular-algorithms-for-modeling-auto-insurance-pure-premium

[^4_6]: https://www.geeksforgeeks.org/machine-learning/feature-selection-using-decision-tree/

[^4_7]: https://www.lincs.fr/wp-content/uploads/2025/02/Feature-Selection-using-Tree-based-Models.pdf

[^4_8]: http://ethen8181.github.io/machine-learning/trees/monotonic.html

[^4_9]: https://www.youtube.com/watch?v=EgKAGrvF8rs

[^4_10]: https://lightgbm.readthedocs.io/en/latest/Parameters.html

[^4_11]: https://www.diva-portal.org/smash/get/diva2:1669166/FULLTEXT01

[^4_12]: https://blog.truegeometry.com/tutorials/education/821592f928acc1c71bb481f719718657/JSON_TO_ARTCL_Gradient_Boosting_for_Pricing_in_context_of_Machine_Learning_for_P.html

[^4_13]: https://www.cs.cornell.edu/~kilian/papers/gbfs.pdf

[^4_14]: https://www.youtube.com/watch?v=JdoBdTlKedA

[^4_15]: https://www.linkedin.com/posts/trijotsingh_insurancepricing-gbm-actuarialscience-activity-7313392706708242433-Xl5h

[^4_16]: http://uc-r.github.io/gbm_regression

[^4_17]: https://www.forbes.com/councils/forbestechcouncil/2024/10/28/harnessing-ai-and-gradient-boosting-for-insurance-premium-modeling/

[^4_18]: http://proceedings.mlr.press/v108/han20a/han20a.pdf

[^4_19]: https://www.reddit.com/r/rstats/comments/1fst92f/what_possible_methods_are_there_for_feature/


---

# Let's go back to the previously suggested GAM/GA2M model on the residuals of the GLM model. Can you choose some example features (35) to explain how we model GLM with say 25 highly correlated features and then how we go ahead and actually model the GAM/GA2M to capture the residual from GLM in terms of non linear/interactions? A simple worked out math is what I'm looking for to understand this better.

Yes, you can model it exactly that way. Below is a toy but fully worked‑through example that shows:

- a baseline **GLM with many (correlated) rating variables**,
- how you define the **residual target**, and
- how a **GAM/GA2M on the residuals** captures non‑linearities and interactions in a fully decomposable way.

I’ll use a small subset of features so the math is readable, but you can imagine these are 5–6 out of your 25 GLM features plus some of the remaining 10 features. The functional form matches standard GLM/GAM/GA2M definitions from the literature.[^5_1][^5_2][^5_3][^5_4][^5_5]

***

## Step 1 – Example feature universe (35) and GLM subset (25)

Suppose you have 35 engineered features for homeowners:

- Core “rating” style features (used in GLM, 25 of them; I’ll show 6):
    - $x_1$: Roof age (years)
    - $x_2$: Territory\_HighWind (0/1)
    - $x_3$: Construction quality score (1–5)
    - $x_4$: Prior claims count (0,1,2,…)
    - $x_5$: Protection class (1–10; higher = worse)
    - $x_6$: Coverage A amount (in 100k units)
    - … plus 19 more (age of dwelling, occupancy, heating type, etc.).
- Additional granular features (not in GLM today, from your remaining 10):
    - $z_1$: Hail exposure index (0–1)
    - $z_2$: Roof condition score from imagery (0–1, 1 = perfect)
    - $z_3$: Wildfire fuel index (0–1)
    - $z_4$: Distance to coast (km)
    - $z_5$: Non‑weather water risk score (0–1)
    - … plus 5 more.

GLM uses the first 25 (including $x_1$–$x_6$) because of regulatory familiarity, collinearity constraints, etc. GAM/GA2M will work on **residuals** and is allowed to use some of these plus the extra $z$’s to pick up missed non‑linearities/interactions.

***

## Step 2 – Baseline GLM (log‑link, multiplicative rating)

Assume a pure‑premium GLM with log link (standard in pricing so factors multiply).[^5_3][^5_1]

$$
\log \mu_{\text{GLM}}(x) \;=\; \beta_0 
+ \beta_1 \cdot \text{RoofAge10} 
+ \beta_2 \cdot \text{Territory\_HighWind}
+ \beta_3 \cdot \text{LowQuality}
+ \beta_4 \cdot \text{PriorClaims}
+ \beta_5 \cdot \text{ProtClassOver5}
+ \ldots
$$

Where we define simple transforms of our GLM features:

- $\text{RoofAge10} = x_1 / 10$ (roof age in decades)
- $\text{Territory\_HighWind} = x_2$
- $\text{LowQuality} = 5 - x_3$ (higher when construction quality is worse)
- $\text{PriorClaims} = x_4$
- $\text{ProtClassOver5} = \max(0, x_5 - 5)$

Pick **illustrative** coefficients (just for math, not from data):

- $\beta_0 = \log(500)$   (baseline pure premium 500)
- $\beta_1 = 0.08$        (per decade of roof age)
- $\beta_2 = 0.30$        (high‑wind territory surcharge)
- $\beta_3 = 0.10$        (per unit lower quality)
- $\beta_4 = 0.25$        (per prior claim)
- $\beta_5 = 0.05$        (per protection‑class step above 5)

So:

$$
\log \mu_{\text{GLM}}(x)
= \log(500)
+ 0.08 \cdot \text{RoofAge10}
+ 0.30 \cdot \text{Territory\_HighWind}
+ 0.10 \cdot \text{LowQuality}
+ 0.25 \cdot \text{PriorClaims}
+ 0.05 \cdot \text{ProtClassOver5}
$$

***

## Step 3 – One concrete policy through the GLM

Take a specific policy (one row):

- Roof age: $x_1 = 25$ years
- Territory\_HighWind: $x_2 = 1$
- Construction quality: $x_3 = 3$ (mid‑quality)
- Prior claims: $x_4 = 1$
- Protection class: $x_5 = 7$

Derived GLM inputs:

- $\text{RoofAge10} = 25/10 = 2.5$
- $\text{Territory\_HighWind} = 1$
- $\text{LowQuality} = 5 - 3 = 2$
- $\text{PriorClaims} = 1$
- $\text{ProtClassOver5} = 7 - 5 = 2$

Plug into GLM:

$$
\begin{aligned}
\log \mu_{\text{GLM}} 
&= \log(500)
+ 0.08 \cdot 2.5
+ 0.30 \cdot 1
+ 0.10 \cdot 2
+ 0.25 \cdot 1
+ 0.05 \cdot 2 \\
&= \log(500)
+ 0.20 + 0.30 + 0.20 + 0.25 + 0.10 \\
&= \log(500) + 1.05
\end{aligned}
$$

So the GLM’s **multiplicative factor** relative to baseline is $\exp(1.05) \approx 2.86$, giving:

$$
\mu_{\text{GLM}} \approx 500 \times 2.86 \approx 1{,}430
$$

Interpretation: based on the 25 GLM features, we price this risk at about 1,430 units of pure premium.

***

## Step 4 – Define the residual the GAM/GA2M will learn

Suppose on held‑out data (or long‑run experience) this kind of risk actually behaves like ~1,800 units of pure premium on average.

That means we want a model that, **on top of the GLM**, learns an uplift:

$$
\text{True } \mu \approx 1{,}800 \quad\Rightarrow\quad
\frac{\text{True } \mu}{\mu_{\text{GLM}}}
\approx \frac{1{,}800}{1{,}430} \approx 1.26
$$

So on the **log scale**, the residual we want to model is:

$$
\delta(x) \;=\; \log(\text{True } \mu) - \log \mu_{\text{GLM}} 
\approx \log(1.26) \approx 0.23
$$

This is exactly the residual that the GAM/GA2M will be trained to predict:

- Training target: $\delta(x) = \log(Y) - \log \hat{\mu}_{\text{GLM}}(x)$ (or equivalently, log of observed/GLM ratio).
- Final prediction:

$$
\log \mu_{\text{final}}(x) = \log \mu_{\text{GLM}}(x) + \hat{\delta}_{\text{GAM}}(x)
$$

This “GLM as offset + additive residual” setup is consistent with how GAMs are used as flexible extensions of GLMs in insurance.[^5_6][^5_7][^5_3]

***

## Step 5 – Structure of the residual GAM/GA2M

Now we build a residual GA2M (GAM + pairwise interactions) on $\delta(x)$.[^5_2][^5_8][^5_5][^5_3]

We choose features where we expect non‑linear or interaction effects the GLM under‑captures, e.g.:

- Hail exposure index $z_1$ (0–1)
- Roof age $x_1$ (again, but now with a flexible curve instead of just linear)
- Prior claims $x_4$ (to allow saturation effects)
- Wildfire fuel index $z_3$
- Interaction between hail exposure and roof age $(z_1, x_1)$

Define the residual GA2M as:

$$
\delta(x) \approx f_1(z_1) + f_2(x_1) + f_3(x_4) + f_{1,2}(z_1, x_1)
$$

- $f_1$ is a 1‑D function of hail exposure index.
- $f_2$ is a 1‑D function of roof age.
- $f_3$ is a 1‑D function of prior claims.
- $f_{1,2}$ is a 2‑D interaction function of hail exposure and roof age.

In a real GA2M, each $f$ is learned as a smooth curve or 2‑D surface from data, but conceptually they sum to the residual.[^5_8][^5_5][^5_2][^5_3]

***

## Step 6 – A tiny “rating table” example for the residual GA2M

To make it concrete, imagine the learned functions (greatly simplified, piecewise) look like this:

### 6.1 Univariate residual functions

1. Hail exposure $z_1$ (0–1):

$$
f_1(z_1) =
\begin{cases}
0.00, & z_1 < 0.3 \\
0.05, & 0.3 \le z_1 < 0.7 \\
0.10, & z_1 \ge 0.7
\end{cases}
$$

2. Roof age $x_1$ (years):

$$
f_2(x_1) =
\begin{cases}
0.00, & x_1 \le 10 \\
0.05, & 10 < x_1 \le 20 \\
0.08, & x_1 > 20
\end{cases}
$$

3. Prior claims count $x_4$:

$$
f_3(x_4) =
\begin{cases}
0.00, & x_4 = 0 \\
0.03, & x_4 = 1 \\
0.06, & x_4 \ge 2
\end{cases}
$$

### 6.2 Pairwise interaction function

4. Interaction between high hail exposure and old roof:

$$
f_{1,2}(z_1, x_1) =
\begin{cases}
0.02, & z_1 \ge 0.7 \text{ and } x_1 > 20 \\
0.00, & \text{otherwise}
\end{cases}
$$

This captures the idea that **old roofs in high‑hail areas** are worse than you’d expect by just adding a linear roof‑age effect and a simple hail zone factor—exactly the kind of structure GA2Ms are designed to capture while still being interpretable.[^5_5][^5_2][^5_8]

***

## Step 7 – Apply GA2M residual model to the same policy

For our example policy, add the extra features:

- Hail exposure index: $z_1 = 0.8$ (high)
- Roof age: $x_1 = 25$ (already know > 20)
- Prior claims: $x_4 = 1$

Look up each function:

- $f_1(z_1 = 0.8) = 0.10$ (high hail)
- $f_2(x_1 = 25) = 0.08$ (old roof)
- $f_3(x_4 = 1) = 0.03$ (one prior claim)
- $f_{1,2}(z_1=0.8, x_1=25) = 0.02$ (bad combo: high hail \& old roof)

So the **predicted residual on the log scale** is:

$$
\hat{\delta}_{\text{GAM}} 
= 0.10 + 0.08 + 0.03 + 0.02 
= 0.23
$$

Exactly the extra 0.23 we wanted to bring GLM’s 1,430 up to about 1,800.

***

## Step 8 – Combine GLM and GA2M residual to get final premium

We now build the final predictor:

$$
\log \mu_{\text{final}}(x)
= \log \mu_{\text{GLM}}(x) + \hat{\delta}_{\text{GAM}}(x)
= [\log(500) + 1.05] + 0.23
$$

Thus the total uplift factor relative to the 500 baseline is:

$$
\exp(1.05 + 0.23) = \exp(1.28) \approx 3.6
$$

Hence:

$$
\mu_{\text{final}}(x) \approx 500 \times 3.6 \approx 1{,}800
$$

And you can **attribute the extra factor $\exp(0.23) \approx 1.26$** exactly:

- Base GLM (all 25 GLM features): factor $\exp(1.05) \approx 2.86$
- Residual GA2M (non‑linear \& interaction effects): factor $\exp(0.23) \approx 1.26$
    - Of that 0.23 on the log scale:
        - +0.10 from high hail exposure $f_1$
        - +0.08 from old roof $f_2$
        - +0.03 from prior claim effect curvature $f_3$
        - +0.02 from the “old roof × high hail” interaction $f_{1,2}$

That is exactly the “glass‑box” decomposition: **every bit of extra variance explained beyond the GLM is mapped to specific 1‑D or 2‑D functions of your features.**[^5_9][^5_2][^5_3][^5_8][^5_5]

***

If you like, next we can:

- Extend this toy to explicitly show how you’d handle **many correlated GLM features** (and concurvity checks / penalties in the GAM).[^5_3]
- Or translate this math into an R/Python‑style pseudo‑workflow (mgcv / pyGAM / InterpretML EBM) that mirrors what you’d actually code.
<span style="display:none">[^5_10]</span>

<div align="center">⁂</div>

[^5_1]: https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf

[^5_2]: https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf

[^5_3]: https://content.naic.org/sites/default/files/call_materials/NAIC%20GAM%20Presentation%20-%20Insurance%20Summit%202021%20-Final.pdf

[^5_4]: https://christophm.github.io/interpretable-ml-book/extend-lm.html

[^5_5]: https://www.fiddler.ai/blog/a-gentle-introduction-to-ga2ms-a-white-box-model

[^5_6]: http://www.scienpress.com/Upload/JSEM/Vol 12_4_1.pdf

[^5_7]: https://kurser.math.su.se/pluginfile.php/20130/mod_folder/content/0/Kandidat/2020/2020_5_report.pdf?forcedownload=1

[^5_8]: https://cran.r-project.org/web/packages/ebm/vignettes/ebm-rjournal.pdf

[^5_9]: https://docs.datarobot.com/latest/en/docs/reference/pred-ai-ref/ga2m.html

[^5_10]: https://docs.datarobot.com/en/docs/modeling/analyze-models/describe/ga2m.html


---

# How do we go about choosing the input features for GAM/GA2M scientifically?

You choose GAM/GA2M inputs almost the same way you’d choose GLM inputs, but with a bit more structure: (1) start from domain‑plausible features, (2) use residual diagnostics to see which ones still carry signal after the GLM, and (3) let penalized GAM / GA2M machinery and cross‑validation decide which main effects and pairwise interactions to keep, while controlling concurvity and complexity.[^6_1][^6_2][^6_3][^6_4][^6_5]

Here’s a concise, “scientific” workflow you can follow.

***

## 1. Start from a principled candidate set

- Include:
    - All features that are **actuarially plausible** for the peril/coverage, including those already in the GLM and the extra ones you’ve engineered (e.g., granular geospatial or imagery features).[^6_6][^6_7]
    - Continuous variables and rich categorical variables where you *expect* non‑linearity (age, amount, hazard scores, etc.).[^6_4][^6_5]
- Exclude up‑front:
    - Features that are not allowed or are borderline from a regulatory/fairness view.

Think of this as defining a **scientific hypothesis space**: “These are all the variables that *could* matter; the GAM/GA2M + selection procedure will tell us which actually do.”

***

## 2. Use residual diagnostics feature‑by‑feature

Given you already have a GLM:

1. Compute residuals (on log scale or deviance) with the GLM as offset.
2. For each candidate feature $x_j$:
    - Fit a **univariate smooth** of residuals vs $x_j$ with offset = GLM prediction (this is just a one‑term GAM):

$$
\delta \sim f_j(x_j) + \text{offset}(\eta_{\text{GLM}})
$$
    - Check:
        - Is the smooth term significant (p‑value, approximate F‑test)?[^6_2][^6_5][^6_4]
        - Does it explain non‑trivial extra deviance / log‑likelihood?
    - Visualize the shape to ensure it’s sensible and not just overfitting noise.[^6_5][^6_4]

Features that show **no clear residual structure** (flat smooths, non‑significant terms) are weak candidates for the GAM layer; features with strong, stable shapes are strong candidates.

This is essentially univariate screening under the GAM framework, and is exactly how GAM tutorials in R (mgcv) suggest exploring candidate smooths.[^6_8][^6_4][^6_5]

***

## 3. Fit a full residual GAM with penalization and let it select

Next, fit a multivariate residual GAM:

$$
\delta \sim \sum_j f_j(x_j) + \text{offset}(\eta_{\text{GLM}})
$$

using:

- **Penalized smoothing splines** for each $f_j$,
- Automatic smoothing parameter selection via GCV/REML or cross‑validation,
- Optionally, extra penalties or group‑lasso style regularization so unneeded smooths shrink to (near) zero.[^6_3][^6_2][^6_4][^6_5]

Feature selection here is “embedded”:

- If a feature does not help once others are present, its smooth gets heavily penalized and becomes (almost) flat, effectively dropping it.[^6_2][^6_4]
- You can further drop terms whose effective degrees of freedom are ≈1 with tiny contribution to deviance explained.

There are explicit studies of **feature selection algorithms in GAMs under concurvity**, comparing stepwise, boosting, regularization, and concurvity‑aware methods—penalization + GCV/REML comes out as a robust baseline.[^6_2]

***

## 4. Control concurvity (nonlinear multicollinearity)

Because you have many correlated features, you need to check **concurvity** (GAM’s analogue of multicollinearity).[^6_9][^6_10][^6_2]

- Use concurvity diagnostics (e.g., mgcv’s `concurvity()`):
    - High indices (near 1) mean one smooth term can be almost perfectly explained by others, making its effect unstable and hard to interpret.[^6_11][^6_10][^6_9]
- Mitigate by:
    - Dropping or combining highly redundant features.
    - Preferring the feature that’s more interpretable / stable across time.
    - Re‑fitting after pruning to ensure shapes are still reasonable.[^6_8][^6_2]

This gives your feature set a more “orthogonal” structure so the remaining smooths are identifiable and easier to explain.

***

## 5. Selecting interactions for GA2M

For GA2M, you also need to choose **which pairs** get interaction terms:

- Start with a GAM that has **only main effects** (univariate smooths).
- Then use a **ranking procedure** for interactions:
    - Caruana’s GA2M paper proposes FAST, a method that efficiently ranks all candidate feature pairs by potential interaction strength, then adds only the top few pairs to the model.[^6_12][^6_13][^6_14][^6_1]
    - In practice, GA2M/EBM tooling does something similar: it trains main effects first, then adds interactions that provide the largest marginal reduction in loss, subject to a cap K.[^6_15][^6_16][^6_3]

Scientifically:

- Measure for each pair $(i,k)$: **drop in deviance / cross‑validated error** when you add $f_{ik}(x_i,x_k)$ on top of existing main effects.
- Keep only a small number (e.g., top 5–20) of interactions that yield clear, stable improvement and have shapes that are interpretable.[^6_16][^6_1][^6_3]
- Impose **hierarchy**: only allow an interaction if at least one (or both) of its main effects is in the model; this is standard in GA2M and keeps things interpretable.[^6_14][^6_1][^6_3]

***

## 6. Validate stability and interpretability

For any selected feature / interaction:

- Check **stability across folds / time periods**: shapes and importance rankings should not swing wildly between samples.
- Sanity‑check shapes against domain knowledge (e.g., risk vs hazard index should be non‑decreasing; if not, you may impose monotonicity constraints).[^6_6][^6_3][^6_5]
- Ensure the final model’s global and local effects can be plotted and explained to non‑technical stakeholders—one of the main selling points of GA2Ms.[^6_17][^6_1][^6_3][^6_16]

If a feature passes “statistical significance + stability + business sense”, it stays; otherwise you drop it and re‑fit.

***

If you’d like, in a next step I can turn this into a concrete pseudo‑code workflow (e.g., using `mgcv` + a GA2M/EBM library) that shows exactly what you would run on your GLM residuals to perform this kind of structured feature and interaction selection.
<span style="display:none">[^6_18][^6_19]</span>

<div align="center">⁂</div>

[^6_1]: https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf

[^6_2]: https://unipub.lib.uni-corvinus.hu/7707/1/s00180-022-01292-7.pdf

[^6_3]: https://www.emergentmind.com/topics/generalized-additive-model-with-pairwise-interactions-ga-m

[^6_4]: https://christophm.github.io/interpretable-ml-book/extend-lm.html

[^6_5]: https://content.naic.org/sites/default/files/call_materials/NAIC%20GAM%20Presentation%20-%20Insurance%20Summit%202021%20-Final.pdf

[^6_6]: https://www.celent.com/insights/429021251

[^6_7]: https://kurser.math.su.se/pluginfile.php/20130/mod_folder/content/0/Kandidat/2020/2020_5_report.pdf?forcedownload=1

[^6_8]: https://m-clark.github.io/generalized-additive-models/issues.html

[^6_9]: https://stat.ethz.ch/R-manual/R-devel/RHOME/library/mgcv/html/concurvity.html

[^6_10]: https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/concurvity.html

[^6_11]: https://campus.datacamp.com/courses/nonlinear-modeling-with-generalized-additive-models-gams-in-r/interpreting-and-visualizing-gams?ex=11

[^6_12]: https://www.microsoft.com/en-us/research/publication/accurate-intelligible-models-pairwise-interactions/

[^6_13]: https://liner.com/review/accurate-intelligible-models-with-pairwise-interactions

[^6_14]: https://dspace-test.anu.edu.au/items/e2ef794c-d527-4702-afe8-f31d9751a9a9

[^6_15]: https://www.addactis.com/blog/explainable-boosting-machine-model-car-insurance/

[^6_16]: https://cran.r-project.org/web/packages/ebm/vignettes/ebm-rjournal.pdf

[^6_17]: https://www.fiddler.ai/blog/a-gentle-introduction-to-ga2ms-a-white-box-model

[^6_18]: https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/blackbox-guided-generalised-linear-model-building-with-nonlife-pricing-applications/5B4935DF7892FA462F26DA48561BA9D7

[^6_19]: https://arxiv.org/html/2311.07597v2

