import React from 'react';

const HeatmapOverlay = ({ imageSrc, heatmapSrc }) => {
  return (
    <div className="relative">
      <img src={imageSrc} alt="original" />
      <img src={heatmapSrc} alt="heatmap" className="absolute top-0 opacity-50" />
    </div>
  );
};

export default HeatmapOverlay;
