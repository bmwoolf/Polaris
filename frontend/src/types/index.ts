export interface FormData {
  experimentName: string;
  targetSpecies: string;
  targetOrgan: string;
  regionSubregion: string;
  targetProteinGene: string;
  primaryAtlas: string;
  atlasMatchingTool: boolean;
  capsidFasta: File | null;
  knownTropismData: File | null;
  promoters: string[];
  promoterMetadata: File | null;
  barcodeFile: File | null;
  simulationMethod: string;
  outputMetrics: string[];
  uncertaintyEstimation: boolean;
}

export interface SimulationResult {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  results?: {
    heatmaps: string[];
    umaps: string[];
    syntheticCounts: string;
    cellxGeneSession: string;
  };
  error?: string;
} 