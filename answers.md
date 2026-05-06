# Tutorial 9 – Tutorial Questions: Answers

## Question 1
**Why is the idea of "previous time steps" important in an LSTM?**

An LSTM (Long Short-Term Memory) network is designed to learn from *sequences*, where the order and content of earlier observations influence what comes next.  
"Previous time steps" provide the temporal context the model needs to detect patterns such as:

- Repeated behaviour (e.g. a customer who bought online three times in a row is likely to buy online again).
- Gradual trends or cycles (e.g. spending rising towards the weekend).

Without previous time steps the model would see only one isolated row and have no information about how behaviour has been changing. The LSTM's gating mechanism (forget, input, output gates) selectively retains important past information in its **cell state** and discards irrelevant information, so longer-range dependencies can be captured without the gradient-vanishing problem that affects plain RNNs.

---

## Question 2
**A medical organisation wants to predict whether a patient might be susceptible to a heart attack. Their model was trained on previous labelled data; 10 features were used as predictors. Which is most suitable – RF or LSTM?**

**Recommended model: Random Forest (RF)**

Reasons:
- Each patient can be represented as **one independent row** of 10 feature values (e.g. age, blood pressure, cholesterol, BMI, smoking status). The prediction is based on a *snapshot* of the patient's current state, not a sequence over time.
- RF handles a mix of numerical and categorical features naturally and does not require the data to be ordered.
- RF is robust to outliers and missing values, which are common in medical datasets.
- RF provides **feature importance** scores, helping clinicians understand which risk factors matter most – a valuable property in healthcare.
- An LSTM would be unnecessary here because the historical ordering of previous *other patients'* rows is not meaningful; each row represents a separate individual.

---

## Question 3
**The hospital wants to predict how a patient's condition will change over the next 7 days, and has the patient's condition recorded for the previous 14 days. Which is most suitable – RF or LSTM?**

**Recommended model: LSTM**

Reasons:
- The data is a **time series**: daily measurements of the same patient over 14 consecutive days. The *order* in which readings occur carries critical information (e.g. a rising fever or declining oxygen saturation over several days is a strong signal).
- An LSTM can learn these temporal dependencies – it remembers which earlier readings are still relevant via its cell state and can model multi-day trends.
- RF treats each row independently; it cannot inherently capture that reading on Day 5 influences Day 6 which influences Day 7, and so on.
- Predicting 7 future days is a **multi-step sequence forecasting** task, for which LSTM-based architectures (with an encoder-decoder or rolling prediction approach) are well-suited.
- In contrast, a Random Forest would need extensive hand-crafted lag features to approximate the same temporal reasoning that an LSTM learns automatically.

---

## Question 4
**Describe the "cell state" in an LSTM. How does it differ from the "hidden state" in terms of functionality and information flow?**

| Property | Cell State (Cₜ) | Hidden State (hₜ) |
|---|---|---|
| **Purpose** | Long-term memory "conveyor belt" that carries information across many time steps | Short-term/working memory output at each time step |
| **Modification** | Updated gently via additive operations (forget gate removes old info, input gate adds new info) | Computed as a filtered, squashed version of the cell state via the output gate |
| **Information flow** | Flows largely unchanged across time steps unless gates actively modify it; gradient flows well (avoids vanishing gradient) | Passed to the next time step *and* used as the prediction output; more heavily transformed |
| **Analogy** | A long-running notebook that records important facts and erases irrelevant ones | A message that summarises what the notebook says right now and is sent to the next step |

In practical terms: the cell state preserves context that may be needed many steps later (e.g. "the patient had surgery three days ago"), while the hidden state is the immediate representation the model uses to make each step's prediction.

---

## Question 5
**What are the three main gates in an LSTM cell? Explain the function of each gate and the activation function typically used.**

### 1. Forget Gate (fₜ)
- **Function:** Decides *what information to discard* from the previous cell state. It looks at the previous hidden state (hₜ₋₁) and the current input (xₜ) and outputs a value between 0 (completely forget) and 1 (completely keep) for each element of the cell state.
- **Activation:** **Sigmoid (σ)** — outputs values in [0, 1], acting as a soft on/off switch.

### 2. Input Gate (iₜ) and Candidate Values (C̃ₜ)
- **Function:** Controls *what new information to add* to the cell state. The input gate (sigmoid) decides *how much* new information to let through; a `tanh` layer creates candidate values representing *what* could be added.
- **Activation:** Input gate uses **Sigmoid (σ)**; the candidate cell state uses **tanh** (outputs in [−1, 1], providing scaled new information).

### 3. Output Gate (oₜ)
- **Function:** Determines *what to output* as the hidden state hₜ. It filters the (updated) cell state through a `tanh` to squash it, then uses the output gate (sigmoid) to select which parts of that filtered cell state to expose as the hidden state/output.
- **Activation:** Gate itself uses **Sigmoid (σ)**; the cell state being filtered uses **tanh**.

**Summary table:**

| Gate | Role | Activation |
|---|---|---|
| Forget | Erase irrelevant past information from cell state | Sigmoid |
| Input | Write new relevant information into cell state | Sigmoid (gate) + tanh (candidate values) |
| Output | Read from cell state to produce the hidden state / prediction | Sigmoid (gate) + tanh (cell state filter) |
