import { useEffect, useRef, useState } from "react";
import DropdownIcon from "../assets/dropdown-icon.svg";

const ChooseBrushType = () => {
  const [showBrushType, setShowBrushType] = useState(false);
  const [selectedBrush, setSelectedBrush] = useState("");
  const dropdownRef = useRef(null);

  const toggleDropdown = (brush) => {
    setShowBrushType(false);
    setSelectedBrush(brush);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowBrushType(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [dropdownRef]);

  const brushTypes = [
    "cloth",
    "neck",
    "necklace",
    "earring",
    "hat",
    "hair",
    "lower lip",
    " upper lip",
    "mouth",
    " right ear",
    "left ear",
    "right eyebrow",
    "left eyebrow",
    "right eye",
    "left eye",
    "eyeglasses",
    "nose",
    "skin",
  ];

  return (
    <div className="relative w-[280px]" ref={dropdownRef}>
      <div
        className={`w-full lg:mx-0 mx-auto flex items-center justify-between gap-3 bg-primaryText pt-1.5 pb-2 px-5 rounded-full cursor-pointer hover:bg-opacity-90 duration-200 ${
          showBrushType === true ? "bg-opacity-90" : "bg-opacity-100"
        }`}
        onClick={() => setShowBrushType((prev) => !prev)}
      >
        <p className="text-xl md:text-2xl font-extrabold text-primaryLightPink">
          {selectedBrush !== "" ? selectedBrush : "Select Brush Type "}{" "}
        </p>

        <img
          src={DropdownIcon}
          alt="dropdown-icon"
          className={`${
            showBrushType === true ? "rotate-180 pt-0" : "rotate-0 pt-2"
          } duration-200`}
        />
      </div>

      <div
        className={`w-full h-32 bg-transparent absolute left-0 px-3.5 z-[9999] ${
          showBrushType === true
            ? "opacity-100 visible bottom-[800px]"
            : "opacity-0 invisible bottom-[780px]"
        } duration-300`}
      >
        {brushTypes?.map((item, idx) => (
          <div
            onClick={() => toggleDropdown(item)}
            key={idx}
            className="w-full text-primaryText bg-primaryLightPink text-center border-b border-b-primaryText cursor-pointer hover:bg-primaryText hover:text-primaryLightPink hover:border-b-primaryLightPink duration-150"
          >
            <p className="py-2 text-2xl font-semibold">{item}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChooseBrushType;
