import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { ensureAuthenticated, launchAuthenticatedContext, saveStorageState } from "../auth.js";
import { config } from "../config.js";

const repoRoot = path.resolve(fileURLToPath(new URL(".", import.meta.url)), "../../..");

type EffectDelta = {
  label: string;
  intensity_option: string;
  p_value: number;
  comment: string;
};

type MappingFault = {
  study_id: string;
  kind: string;
  evidence_factory_id: number | null;
};

type ModelPlan = {
  study_id: string;
  evidence_factory_id: number;
  deltas: EffectDelta[];
  unmatched_local_effects: string[];
  extra_effect_nodes: string[];
  incomplete_effects: string[];
  ambiguous_local_effects: string[];
};

type PlanPayload = {
  apply_allowed: boolean;
  faults: MappingFault[];
  plans: ModelPlan[];
};

function pythonSync(args: string[], input?: string): string {
  return execFileSync("uv", ["run", "python", "-m", "src.evidence_editor_sync", ...args], {
    cwd: repoRoot,
    encoding: "utf8",
    input,
    maxBuffer: 32 * 1024 * 1024,
  });
}

function printFaults(faults: MappingFault[]): void {
  console.error("Apply refused: mapping integrity faults");
  for (const fault of faults) {
    const extra = fault.evidence_factory_id == null ? "" : ` id=${fault.evidence_factory_id}`;
    console.error(`  ${fault.study_id} ${fault.kind}${extra}`);
  }
}

async function applyPlans(plans: ModelPlan[]): Promise<void> {
  const { browser, context, page } = await launchAuthenticatedContext();
  try {
    await page.goto(config.baseUrl, { waitUntil: "domcontentloaded" });
    await ensureAuthenticated(page, context);
    await saveStorageState(context);

    for (const plan of plans) {
      if (plan.deltas.length === 0) continue;
      const dataUrl = `${config.baseUrl}/evidenceEditor/evidencedata?evidenceId=${plan.evidence_factory_id}`;
      const getResponse = await page.request.get(dataUrl);
      if (!getResponse.ok()) {
        throw new Error(`Failed to read evidence ${plan.evidence_factory_id}: HTTP ${getResponse.status()}`);
      }
      const dto = await getResponse.json();
      const patched = JSON.parse(
        pythonSync(["patch-dto"], JSON.stringify({ dto, deltas: plan.deltas })),
      ) as unknown;
      const postResponse = await page.request.post(`${config.baseUrl}/evidenceEditor`, {
        data: [patched],
        headers: { "content-type": "application/json" },
      });
      if (!postResponse.ok()) {
        throw new Error(
          `Failed to save ${plan.study_id} editor ${plan.evidence_factory_id}: HTTP ${postResponse.status()}`,
        );
      }
      console.log(
        `Wrote ${plan.deltas.length} delta(s) on ${plan.study_id} editor ${plan.evidence_factory_id}`,
      );
    }
  } finally {
    await browser.close();
  }
}

async function main(): Promise<void> {
  const command = process.argv[2];
  if (command !== "plan" && command !== "apply") {
    console.error("usage: npm run editor:plan | npm run editor:apply");
    process.exitCode = 2;
    return;
  }
  if (command === "plan") {
    process.stdout.write(pythonSync(["plan"]));
    return;
  }

  const payload = JSON.parse(pythonSync(["plan", "--json"])) as PlanPayload;
  if (!payload.apply_allowed) {
    printFaults(payload.faults);
    process.exitCode = 2;
    return;
  }
  await applyPlans(payload.plans);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
