class PlannerAgent:

    def choose_model(self, dataset_name):

        if dataset_name == "HIV":
            return "Random Forest"

        elif dataset_name == "PAMPA":
            return "Logistic Regression"

        elif dataset_name == "DAVIS":
            return "Random Forest"

        else:
            return "Random Forest"


planner = PlannerAgent()

dataset = input("Enter dataset (HIV/PAMPA/DAVIS): ")

model = planner.choose_model(dataset)

print("\n===== PLANNER RESULT =====")
print("Dataset:", dataset)
print("Selected Model:", model)
print("==========================")