import { useMemo } from "react";
import { RJSFSchema } from "@rjsf/utils";
import { useGetV2ListModels } from "@/app/api/__generated__/endpoints/llm/llm";
import type { LlmModelsResponse } from "@/app/api/__generated__/models/llmModelsResponse";
import { LlmModelMetadata, LlmModelMetadataMap } from "./types";

type LlmModelSchema = RJSFSchema & {
  llm_model_metadata?: LlmModelMetadataMap;
};

export function useLlmModelField(schema: RJSFSchema, formData: unknown) {
  const { data: registryData } = useGetV2ListModels(
    { enabled_only: false },
    { query: { staleTime: 60_000 } },
  );

  const schemaMetadata = useMemo(
    () => (schema as LlmModelSchema)?.llm_model_metadata ?? {},
    [schema],
  );

  // Merge live is_enabled / is_recommended flags from the registry into the
  // static schema metadata so the picker reflects admin changes without a
  // server restart. Models absent from the registry are hidden.
  const models = useMemo<LlmModelMetadata[]>(() => {
    const responseData = registryData?.data;
    const registryModels: LlmModelsResponse["models"] =
      responseData && "models" in responseData ? responseData.models : [];
    const bySlug = new Map(registryModels.map((m) => [m.slug, m]));

    return Object.values(schemaMetadata)
      .map((m) => {
        const live = bySlug.get(m.name);
        return {
          ...m,
          // Registry is the sole source of truth; absent = not shown.
          is_enabled: live?.is_enabled ?? false,
          is_recommended: live?.is_recommended ?? false,
        };
      })
      .filter((m) => m.is_enabled !== false);
  }, [schemaMetadata, registryData]);

  const selectedName =
    typeof formData === "string"
      ? formData
      : typeof schema.default === "string"
        ? schema.default
        : "";

  const selectedModel = selectedName
    ? models.find((m) => m.name === selectedName)
    : undefined;

  // Registry flag takes priority; schema.default is the static fallback.
  const recommendedModel =
    models.find((m) => m.is_recommended) ??
    models.find(
      (m) => typeof schema.default === "string" && m.name === schema.default,
    );

  return { models, selectedModel, recommendedModel };
}
