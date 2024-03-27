import { useState } from "react";

const ImageUploader = () => {
  const [photoUpload, setPhotoUpload] = useState(null);

  const handlePhotoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setPhotoUpload(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="w-full flex items-center justify-center my-10 md:my-0">
      {photoUpload ? (
        <div>
          <img
            src={photoUpload}
            alt="Uploaded"
            className="w-[350px] object-cover"
          />
        </div>
      ) : (
        <div>
          <label
            htmlFor="photo-upload"
            className="bg-primaryText text-primaryLightPink px-6 pt-1.5 pb-2 md:text-2xl xl:text-[32px] font-extrabold leading-normal rounded-full cursor-pointer"
          >
            Try Now
          </label>

          <input
            id="photo-upload"
            type="file"
            accept="image/*"
            onChange={handlePhotoUpload}
            style={{
              display: "none",
            }}
          />
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
