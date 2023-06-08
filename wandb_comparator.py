"""
Let's create a GitHub Actions workflow that:

1. Acts under `/wandb-comparator <run_id>` (*chatops*) style comments.
2. Takes the information of the run with tag "baseline" of our project in W&B (W&B API)
3. Generate a report comparing the baseline with the `<run_id>` (W&B API)
4. Comment on the Pull Request with the URL with the comparison report (*chatops*)
"""
import os
import wandb
import wandb.apis.reports as wr
from ghapi.core import GhApi

assert os.getenv("WANDB_API_KEY"), "You must set the WANDB_API_KEY environment variable"

# 1. Acts under `/wandb-comparator <run_id>` (*chatops*) style comments.
pr_message = os.environ['PR_MESSAGE']
# The message has the "/wandb-comparator <run_id>" format
# check that the message is in the correct format
if not pr_message.startswith('/wandb-comparator'):
    print('Message does not start with /wandb-comparator')
    exit(1)


# 2. Takes the information of the run with tag "baseline" of our project in W&B 
wandb_api = wandb.Api()
WANDB_PROJECT = "Neurovias"
WANDB_ENTITY = "marioparreno"
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
BASELINE_TAG = ["baseline"]

candidate_runs = wandb_api.runs(
    WANDB_PATH,
    {"tags": {"$in": BASELINE_TAG}}  # this is the Mongo Client
)

assert len(candidate_runs) == 1, "There should be only one baseline run"

BASELINE_RUN = candidate_runs[0]


# 3. Generate a report comparing the baseline with the `<run_id>` (W&B API)
# Get the run id from the message
run_id = pr_message.split(' ')[1]

report = wr.Report(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    title="Compare Runs",
    description="Comparin baseline with candidate",
)

pg = wr.PanelGrid(
    runsets=[
        wr.Runset(
            WANDB_ENTITY, WANDB_PROJECT, "Run Comparison"
        ).set_filters_with_python_expr(
            f"Name in ['{BASELINE_RUN.name}', '{run_id}']"
        )
    ],
    panels=[
        wr.RunComparer(diff_only='split', layout={'w': 24, 'h': 15})
    ]
)

report.blocks = report.blocks[:1] + [pg] + report.blocks[1:]
report.save()


# 4. Comment on the Pull Request with the URL with the comparison report (*chatops*)
# Create a comment on the PR with the report
owner,repo = os.environ['REPO'].split('/')
gh_api = GhApi(owner=owner, repo=repo)

pr_number = os.environ['PR_NUMBER']

report_url = f"[in this report]({report.url})"
gh_api.issues.create_comment(
    issue_number=pr_number,
    body=f"A comparison between the linked run and baseline is available {report_url}."
)
