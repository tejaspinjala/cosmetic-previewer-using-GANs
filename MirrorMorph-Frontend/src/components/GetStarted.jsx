import ChooseBrushType from "./ChooseBrushType";
import RangeSlider from "./RangeSlider";
import ImageUploader from "./ImageUploader";

const GetStarted = () => {
  return (
    <div className="w-full max-w-[1215px] mx-auto pb-8 lg:pb-12 px-4">
      <h2 className="text-3xl md:text-[40px] text-primaryText font-extrabold text-center">
        Shall we get started?
      </h2>

      <div className="w-full mt-[94px] flex justify-between xl:flex-row flex-col gap-12">
        <div className="w-full">
          <ChooseBrushType />

          <RangeSlider />

          <div className="flex items-center justify-center lg:justify-start">
            <button className="bg-primaryText text-primaryLightPink px-4 pt-1.5 pb-2 md:text-2xl xl:text-[32px] font-extrabold leading-normal rounded-full hover:translate-y-2 hover:bg-opacity-90 duration-200">
              Change Brush
            </button>
          </div>
        </div>

        <ImageUploader />

        <div className="w-full flex lg:flex-col justify-between items-end">
          <button className="bg-primaryText text-primaryLightPink px-4 pt-1.5 pb-2 md:text-2xl xl:text-[32px] font-extrabold leading-normal rounded-full hover:bg-opacity-90 hover:translate-y-2 duration-200">
            Get Started
          </button>

          <button className="bg-primaryText text-primaryLightPink px-4 pt-1.5 pb-2 md:text-2xl xl:text-[32px] font-extrabold leading-normal rounded-full hover:bg-opacity-90 hover:translate-y-2 duration-200">
            Get Results
          </button>
        </div>
      </div>
    </div>
  );
};

export default GetStarted;
