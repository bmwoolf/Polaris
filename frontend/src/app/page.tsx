'use client';

import { useState } from 'react';
import { ChevronDownIcon } from '@heroicons/react/24/outline';
import { FormData } from '@/types';

export default function Home() {
  const [formData, setFormData] = useState<FormData>({
    experimentName: '',
    targetSpecies: '',
    targetOrgan: '',
    regionSubregion: '',
    targetProteinGene: '',
    primaryAtlas: '',
    atlasMatchingTool: false,
    capsidFasta: null,
    knownTropismData: null,
    promoters: [],
    promoterMetadata: null,
    barcodeFile: null,
    simulationMethod: '',
    outputMetrics: [],
    uncertaintyEstimation: false,
  });

  const handleInputChange = (field: keyof FormData, value: string | boolean | string[]) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleFileChange = (field: keyof FormData, file: File | null) => {
    setFormData(prev => ({ ...prev, [field]: file }));
  };

  const handleMultiSelect = (field: keyof FormData, value: string, checked: boolean) => {
    if (checked) {
      setFormData(prev => ({ 
        ...prev, 
        [field]: [...(prev[field] as string[]), value] 
      }));
    } else {
      setFormData(prev => ({ 
        ...prev, 
        [field]: (prev[field] as string[]).filter(item => item !== value) 
      }));
    }
  };

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitMessage('');

    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      
      if (response.ok) {
        setSubmitMessage(`Simulation started successfully! ID: ${result.id}`);
      } else {
        setSubmitMessage(`Error: ${result.error}`);
      }
    } catch {
      setSubmitMessage('Failed to submit simulation request');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          Computational scAAVengr- v1.0
        </h1>
        
        <hr className="border-gray-700 mb-8" />
        
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* 1. Experiment Setup */}
          <section>
            <h2 className="text-xl font-semibold mb-4">1. Experiment Setup</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Experiment Name
                </label>
                <input
                  type="text"
                  value={formData.experimentName}
                  onChange={(e) => handleInputChange('experimentName', e.target.value)}
                  placeholder="Experiment name"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Target Species
                </label>
                <select
                  value={formData.targetSpecies}
                  onChange={(e) => handleInputChange('targetSpecies', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select species</option>
                  <option value="human">Human</option>
                  <option value="mouse">Mouse</option>
                  <option value="macaque">Macaque</option>
                  {/* <option value="other">Other</option> */}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Target Organ
                </label>
                <select
                  value={formData.targetOrgan}
                  onChange={(e) => handleInputChange('targetOrgan', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select organ</option>
                  <option value="retina">Retina</option>
                  <option value="brain">Brain</option>
                  {/* <option value="other">Other</option> */}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Region / Subregion
                </label>
                <select
                  value={formData.regionSubregion}
                  onChange={(e) => handleInputChange('regionSubregion', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select region</option>
                  <option value="photoreceptors">Photoreceptors</option>
                  <option value="hippocampus">Hippocampus</option>
                  {/* <option value="other">Other</option> */}
                </select>
              </div>
              
              <div className="md:col-span-2">
                <label className="block text-sm font-medium mb-2">
                  Target Protein / Gene
                </label>
                <input
                  type="text"
                  value={formData.targetProteinGene}
                  onChange={(e) => handleInputChange('targetProteinGene', e.target.value)}
                  placeholder="Payload they want expressed"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </section>
          
          <hr className="border-gray-700" />
          
          {/* 2. Cell Atlas Selection */}
          <section>
            <h2 className="text-xl font-semibold mb-4">2. Cell Atlas Selection</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Primary Atlas
                </label>
                <input
                  type="text"
                  value={formData.primaryAtlas}
                  onChange={(e) => handleInputChange('primaryAtlas', e.target.value)}
                  placeholder="File upload or Arc browser search"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-sm text-gray-400 mt-1">
                  Choose from Arc Virtual Cell Atlas or upload custom
                </p>
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="atlasMatchingTool"
                  checked={formData.atlasMatchingTool}
                  onChange={(e) => handleInputChange('atlasMatchingTool', e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                />
                <label htmlFor="atlasMatchingTool" className="ml-2 text-sm">
                  Atlas Matching Tool
                </label>
                <p className="text-sm text-gray-400 ml-2">
                  Auto-harmonize custom atlas with Arc reference
                </p>
              </div>
            </div>
          </section>
          
          <hr className="border-gray-700" />
          
          {/* 3. AAV Capsid Input */}
          <section>
            <h2 className="text-xl font-semibold mb-4">3. AAV Capsid Input</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Capsid FASTA Upload
                </label>
                <input
                  type="file"
                  accept=".fasta,.fa"
                  onChange={(e) => handleFileChange('capsidFasta', e.target.files?.[0] || null)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-sm text-gray-400 mt-1">
                  VP protein sequences
                </p>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Known Tropism Data
                </label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => handleFileChange('knownTropismData', e.target.files?.[0] || null)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-sm text-gray-400 mt-1">
                  Optional CSV with prior transduction rates for any capsids
                </p>
              </div>
            </div>
          </section>
          
          <hr className="border-gray-700" />
          
          {/* 4. Promoter Input */}
          <section>
            <h2 className="text-xl font-semibold mb-4">4. Promoter Input</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Promoter List
                </label>
                <div className="space-y-2">
                  {['CAG', 'CMV', 'RHO'].map((promoter) => (
                    <label key={promoter} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={formData.promoters.includes(promoter)}
                        onChange={(e) => handleMultiSelect('promoters', promoter, e.target.checked)}
                        className="w-4 h-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm capitalize">{promoter}</span>
                    </label>
                  ))}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Promoter Metadata Upload
                </label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => handleFileChange('promoterMetadata', e.target.files?.[0] || null)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <p className="text-sm text-gray-400 mt-1">
                  Optional CSV with tissue specificity/strength values
                </p>
              </div>
            </div>
          </section>
          
          <hr className="border-gray-700" />
          
          {/* 5. Barcode Mapping */}
          <section>
            <h2 className="text-xl font-semibold mb-4">5. Barcode Mapping</h2>
            <div>
              <label className="block text-sm font-medium mb-2">
                Barcode File Upload
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => handleFileChange('barcodeFile', e.target.files?.[0] || null)}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-sm text-gray-400 mt-1">
                CSV mapping [Capsid ID, Promoter ID, Barcode Seq]
              </p>
            </div>
          </section>
          
          <hr className="border-gray-700" />
          
          {/* 6. Run Settings */}
          <section>
            <h2 className="text-xl font-semibold mb-4">6. Run Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Simulation Method
                </label>
                <select
                  value={formData.simulationMethod}
                  onChange={(e) => handleInputChange('simulationMethod', e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select method</option>
                  <option value="scDesign3">scDesign3</option>
                  <option value="scVI">scVI</option>
                  <option value="hybrid">Hybrid</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">
                  Output Metrics
                </label>
                <div className="space-y-2">
                  {['heatmaps', 'UMAPs', 'synthetic counts', 'CellxGene session'].map((metric) => (
                    <label key={metric} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={formData.outputMetrics.includes(metric)}
                        onChange={(e) => handleMultiSelect('outputMetrics', metric, e.target.checked)}
                        className="w-4 h-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm">{metric}</span>
                    </label>
                  ))}
                </div>
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="uncertaintyEstimation"
                  checked={formData.uncertaintyEstimation}
                  onChange={(e) => handleInputChange('uncertaintyEstimation', e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500"
                />
                <label htmlFor="uncertaintyEstimation" className="ml-2 text-sm">
                  Uncertainty Estimation
                </label>
                <p className="text-sm text-gray-400 ml-2">
                  Include CV/CI in outputs
                </p>
              </div>
            </div>
          </section>
          
          <hr className="border-gray-700" />
          
          {/* 7. Submit & Run */}
          <section>
            <h2 className="text-xl font-semibold mb-4">7. Submit & Run</h2>
            <div className="text-center">
              <button
                type="submit"
                disabled={isSubmitting}
                className={`font-semibold py-3 px-8 rounded-lg text-lg transition-colors duration-200 ${
                  isSubmitting 
                    ? 'bg-gray-600 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-700'
                } text-white`}
              >
                {isSubmitting ? 'Starting Simulation...' : 'Run Simulation'}
              </button>
              <p className="text-sm text-gray-400 mt-2">
                Kicks off full pipeline
              </p>
              
              {submitMessage && (
                <div className={`mt-4 p-3 rounded-md ${
                  submitMessage.includes('Error') || submitMessage.includes('Failed')
                    ? 'bg-red-900 text-red-200'
                    : 'bg-green-900 text-green-200'
                }`}>
                  {submitMessage}
                </div>
              )}
              
              <div className="flex justify-center mt-4">
                <ChevronDownIcon className="h-6 w-6 text-gray-400" />
              </div>
            </div>
          </section>
        </form>
      </div>
    </div>
  );
}
