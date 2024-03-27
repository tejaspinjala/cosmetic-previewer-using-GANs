import Logo from "../assets/logo.png";
import Section1Image from "../assets/banner-first.jpeg";

const Banner = () => {
  return (
    <div className="w-full">
      <div className="w-full bg-primaryLightPink">
        <div className="w-full flex items-center justify-center">
          <img src={Logo} alt="logo" className="w-[92px]" />
        </div>

        <div className="text-center mt-5 pb-44">
          <h1 className="text-5xl md:text-7xl lg:text-8xl 2xl:text-9xl text-primaryText font-extrabold">
            MirrorMorph
          </h1>
        </div>
      </div>

      <div className="w-full max-w-[1215px] mx-auto px-4">
        <div className="flex items-center justify-center -mt-[155px]">
          <img
            src={Section1Image}
            alt="banner-image"
            className="w-full md:w-[494px] rounded-[546px] object-cover"
          />
        </div>

        <div className="mt-6 text-center">
          <p className="text-xl md:text-2xl lg:text-3xl xl:text-[40px] tracking-[0.8px] text-primaryText leading-normal font-extrabold">
            {'"'}see tomorrow{"'"}s you today: preview your transformation{'"'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Banner;
