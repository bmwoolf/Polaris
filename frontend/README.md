# Computational scAAVengr Frontend

A Next.js application that provides a user interface for the Computational scAAVengr tool, allowing scientists to predict RNA/DNA delivery tropism computationally.

## Features

- **Experiment Setup**: Configure target species, organ, region, and protein/gene
- **Cell Atlas Selection**: Choose from Arc Virtual Cell Atlas or upload custom data
- **AAV Capsid Input**: Upload FASTA files with viral protein sequences
- **Promoter Selection**: Choose from CAG, CMV, RHO, or custom promoters
- **Barcode Mapping**: Upload CSV files mapping capsids, promoters, and barcodes
- **Run Settings**: Configure simulation method and output metrics
- **Real-time Feedback**: Get immediate feedback on simulation status

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Heroicons
- **State Management**: React hooks

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Development Workflow

### Live Reloading
Next.js provides excellent hot reloading out of the box:
- **Fast Refresh**: React components update instantly without losing state
- **CSS Hot Reload**: Tailwind changes apply immediately
- **TypeScript**: Type checking happens in real-time
- **Turbopack**: Faster builds and development experience

### Development Commands
```bash
# Start development server with hot reloading
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Lint code
npm run lint
```

### File Watching
The development server automatically watches for changes in:
- `src/**/*` - All source files
- `public/**/*` - Static assets
- Configuration files (next.config.js, tailwind.config.js)

## API Integration

The frontend is designed to connect to an ensemble of models that will:

1. **Process Cell Data**: Pull retinal + brain single-cell atlases from Arc Virtual Cell Atlas
2. **Generate Cell Embeddings**: Use scVI to encode each cell type
3. **Process Capsid Data**: Use ESM-2/Evo2 on viral protein sequences
4. **Encode Promoters**: Create vectors for CAG, CMV, RHO promoters
5. **Predict Tropism**: Combine vectors to predict transduction rates per cell type
6. **Generate Synthetic Data**: Use scDesign3/scVI to create realistic scRNA-seq data
7. **Build Reference**: Append 8nt AAV barcode sequences to genome
8. **Analyze Results**: Run STARsolo → Scanpy for clustering and visualization

## Form Structure

The UI collects all necessary inputs to replicate the wet-lab scAAVengr pipeline:

- **Input Files**: FASTA sequences, CSV metadata, barcode mappings
- **Parameters**: Species, organ, region, simulation method
- **Output Preferences**: Heatmaps, UMAPs, synthetic counts, CellxGene sessions

## Future Enhancements

- Real-time simulation progress tracking
- Results visualization and download
- Integration with CellxGene for data sharing
- Model performance metrics and validation
- Batch processing capabilities

## Development

- `src/app/page.tsx` - Main form component
- `src/types/index.ts` - TypeScript interfaces
- `src/app/api/simulate/route.ts` - API endpoint for simulation requests
- `src/app/globals.css` - Global styles and Tailwind configuration

## Tips for Development

1. **Use the browser dev tools** to inspect the form state
2. **Check the console** for any errors or form submission data
3. **Modify the UI** - changes will appear instantly in the browser
4. **Test form submission** - the API endpoint logs all received data
5. **Responsive design** - test on different screen sizes
