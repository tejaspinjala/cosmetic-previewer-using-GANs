import SecondSectionImage from "../assets/banner-two.png";

import LineBreak from "./LineBreak";

const Introduction = () => {
  return (
    <div className="w-full max-w-[1215px] mx-auto mt-8 md:mt-16 py-8 md:py-14 px-4">
      <LineBreak />

      <div className="pt-10 md:pt-[70px] pb-14 md:pb-[98px] px-2 md:px-7 flex items-center justify-between md:flex-row flex-col-reverse gap-10 md:gap-20 xl:gap-[150px]">
        <div className="w-full md:max-w-[748px]">
          <p className="md:text-xl lg:text-[22px] xl:text-[25px] font-semibold text-primaryText lg:leading-[50px]">
            Introducing our groundbreaking project: MirrorMorph - where the
            future of your appearance begins with a click. This innovation
            transforms the way you envision cosmetic enhancements, <br />{" "}
            offering a virtual platform for users and surgeons alike to preview
            and experiment with potential facial modifications before making
            life-changing decisions.
          </p>
        </div>

        <div>
          <img src={SecondSectionImage} alt="" />
        </div>
      </div>

      <LineBreak />
    </div>
  );
};

export default Introduction;
