import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import type { vtkImageData as vtkImageDataType } from '@kitware/vtk.js/Common/DataModel/ImageData';

/**
 * Apply image sharpening using Laplacian edge enhancement for 2D images.
 * This function implements edge enhancement by subtracting the Laplacian
 * (edge detection) from the original image to enhance edges.
 *
 * @param imageData - The vtkImageData to sharpen
 * @param intensity - Sharpening intensity (0-3, where 0 is no sharpening, 3 is maximum)
 * @returns The sharpened vtkImageData
 */
function applySharpeningFilter({
  imageData,
  intensity = 0.5,
}: {
  imageData: vtkImageDataType;
  intensity?: number;
}): vtkImageDataType {
  if (!imageData) {
    return imageData;
  }

  // Clamp intensity to valid range
  intensity = Math.max(0, Math.min(3, intensity));

  if (intensity === 0) {
    // No sharpening needed
    return imageData;
  }

  const dims = imageData.getDimensions();
  const scalars = imageData.getPointData().getScalars();
  const data = scalars.getData();
  const numComponents = scalars.getNumberOfComponents();

  // Create output data
  const outputData = new Float32Array(data.length);

  // Optimized: Pre-calculate strides for faster indexing
  const xStride = numComponents;
  const yStride = dims[0] * xStride;
  const zStride = dims[1] * yStride;

  // Process interior pixels (no boundary checks needed)
  for (let z = 0; z < dims[2]; z++) {
    const zOffset = z * zStride;

    // Handle edges separately to avoid boundary checks in main loop
    for (let y = 1; y < dims[1] - 1; y++) {
      const yOffset = zOffset + y * yStride;

      for (let x = 1; x < dims[0] - 1; x++) {
        const xOffset = yOffset + x * xStride;

        // Unrolled component loop for better performance
        if (numComponents === 1) {
          // Most common case: grayscale
          const idx = xOffset;
          const center = data[idx];

          // Optimized Laplacian calculation (only non-zero kernel values)
          const laplacian =
            data[idx - yStride] + // top
            data[idx - xStride] + // left
            data[idx + xStride] + // right
            data[idx + yStride] - // bottom
            4 * center; // center

          outputData[idx] = center - laplacian * intensity;
        } else {
          // Multi-component case
          for (let c = 0; c < numComponents; c++) {
            const idx = xOffset + c;
            const center = data[idx];

            const laplacian =
              data[idx - yStride] +
              data[idx - xStride] +
              data[idx + xStride] +
              data[idx + yStride] -
              4 * center;

            outputData[idx] = center - laplacian * intensity;
          }
        }
      }
    }

    // Handle boundary pixels (copy original values or apply simpler filter)
    // Top and bottom rows
    for (let x = 0; x < dims[0]; x++) {
      const topIdx = zOffset + x * xStride;
      const bottomIdx = zOffset + (dims[1] - 1) * yStride + x * xStride;

      for (let c = 0; c < numComponents; c++) {
        outputData[topIdx + c] = data[topIdx + c];
        outputData[bottomIdx + c] = data[bottomIdx + c];
      }
    }

    // Left and right columns (excluding corners already handled)
    for (let y = 1; y < dims[1] - 1; y++) {
      const leftIdx = zOffset + y * yStride;
      const rightIdx = leftIdx + (dims[0] - 1) * xStride;

      for (let c = 0; c < numComponents; c++) {
        outputData[leftIdx + c] = data[leftIdx + c];
        outputData[rightIdx + c] = data[rightIdx + c];
      }
    }
  }

  // Create new image data with sharpened values
  const outputImageData = vtkImageData.newInstance();
  outputImageData.setDimensions(dims);
  outputImageData.setSpacing(imageData.getSpacing());
  outputImageData.setOrigin(imageData.getOrigin());
  outputImageData.setDirection(imageData.getDirection());

  const outputScalars = vtkDataArray.newInstance({
    numberOfComponents: numComponents,
    values: outputData,
    name: 'Scalars',
  });

  outputImageData.getPointData().setScalars(outputScalars);

  return outputImageData;
}

export { applySharpeningFilter };
