export type LlmModelMetadata = {
  creator: string;
  creator_name: string;
  title: string;
  provider: string;
  provider_name: string;
  name: string;
  price_tier?: number;
  is_recommended?: boolean;
  is_enabled?: boolean;
};

export type LlmModelMetadataMap = Record<string, LlmModelMetadata>;
