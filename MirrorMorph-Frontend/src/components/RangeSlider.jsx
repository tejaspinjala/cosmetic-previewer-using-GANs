import { useState } from "react";

const RangeSlider = () => {
  const [sliderValue, setSliderValue] = useState(1);

  return (
    <div className="w-9/12 my-12 pt-10 lg:pt-32 pb-8 lg:pb-20">
      <input
        type="range"
        className="w-full accent-primaryText border-none outline-none shadow-none focus:border-none focus:outline-none"
        min={1}
        max={10}
        value={sliderValue}
        onChange={(e) => setSliderValue(e.target.value)}
      />

      <p className="text-2xl font-semibold text-primaryText text-center">
        {sliderValue}px
      </p>
    </div>
  );
};

export default RangeSlider;
