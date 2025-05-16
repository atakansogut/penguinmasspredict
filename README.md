This project uses a multiple linear regression model to predict the body mass of penguins based on three key features: bill length, gender, and species. The model is built using the statsmodels library and leverages its formula API to automatically handle categorical variables.

Dataset: Palmer Penguins Dataset (or local CSV if that's what you're using)

Features used: bill_length_mm, gender, species

Target variable: body_mass_g

Missing values were dropped and categorical variables were handled using the C() function in the regression formula.
