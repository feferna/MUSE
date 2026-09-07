import {useState, useRef} from 'react';
import PARAM_CONFIGS from './paramConfigs';

export default function AnimatedToggle({params}) {
  const [isOn, setIsOn] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const toggleRef = useRef(null);

  const handleHover = () => setIsOn(!isOn);

  // Colors - handle both numeric indices and string values
  const scheme = (() => {
    if (typeof params?.colorScheme === 'number') {
      // Handle old index-based system
      return (
        PARAM_CONFIGS.colorScheme.options[params.colorScheme] ||
        PARAM_CONFIGS.colorScheme.options[0]
      );
    } else {
      // Handle new value-based system
      return (
        PARAM_CONFIGS.colorScheme.options.find(
          o => o.value === params?.colorScheme,
        ) || PARAM_CONFIGS.colorScheme.options[0]
      );
    }
  })();

  if (!scheme) return <div>Error: Invalid colorScheme</div>;
  const [activeColor, inactiveColor, thumbColor] = scheme.colors;

  const visualStyleConfig = (() => {
    if (typeof params?.visualStyle === 'number') {
      // Handle old index-based system
      return (
        PARAM_CONFIGS.visualStyle.options[params.visualStyle] ||
        PARAM_CONFIGS.visualStyle.options[0]
      );
    } else {
      // Handle new value-based system
      return (
        PARAM_CONFIGS.visualStyle.options.find(
          o => o.value === params?.visualStyle,
        ) || PARAM_CONFIGS.visualStyle.options[0]
      );
    }
  })();

  const {visual: visualStyle, track: trackStyle} = visualStyleConfig.style;

  // Handle other parameters - thumbShape
  const thumbShape = (() => {
    if (typeof params?.thumbShape === 'number') {
      return (
        PARAM_CONFIGS.thumbShape.options[params.thumbShape]?.value || 'circle'
      );
    } else {
      return params?.thumbShape || 'circle';
    }
  })();

  const easing = (() => {
    if (typeof params?.easing === 'number') {
      return (
        PARAM_CONFIGS.easing.options[params.easing]?.value || 'ease-in-out'
      );
    } else {
      return params?.easing || 'ease-in-out';
    }
  })();

  // Fixed dimensions (since size/shape removed)
  const dimensions = {width: 200, height: 100};

  // Track style
  const getTrackStyle = () => {
    const base = {
      width: `${dimensions.width}px`,
      height: `${dimensions.height}px`,
      borderRadius: `${params.borderRadius}px`,
      position: 'relative',
      cursor: 'pointer',
      transition: `all ${params.duration}ms ${easing}`,
    };
    const trackColor = isOn ? activeColor : inactiveColor;

    switch (trackStyle) {
      case 'gradient':
        return {
          ...base,
          background: `linear-gradient(90deg, ${trackColor}, ${trackColor}dd)`,
        };
      case 'outlined':
        return {
          ...base,
          background: 'transparent',
          border: `2px solid ${trackColor}`,
        };
      case 'textured':
        return {
          ...base,
          background: trackColor,
          backgroundImage:
            'repeating-linear-gradient(45deg, rgba(255,255,255,0.08) 0 4px, transparent 4px 8px)',
        };
      case 'glass':
        return {
          ...base,
          background: `${trackColor}40`,
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.2)',
        };
      default:
        return {...base, background: trackColor};
    }
  };

  // Thumb geometry
  const thumbSize =
    Math.min(dimensions.width, dimensions.height) * params.thumbRatio;
  const padding =
    (Math.min(dimensions.width, dimensions.height) - thumbSize) / 2;
  const maxTranslate = dimensions.width - thumbSize - padding * 2;
  const translate = isOn ? maxTranslate : 0;

  const getThumbShapeStyle = (shape, size) => {
    const base = {width: `${size}px`, height: `${size}px`};
    switch (shape) {
      case 'square':
        return {...base, borderRadius: '0px'};
      case 'diamond': // ✅ FIXED: no transform, so translate works
        return {
          ...base,
          clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)',
        };
      case 'hexagon':
        return {
          ...base,
          clipPath:
            'polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%)',
        };
      case 'star':
        return {
          ...base,
          clipPath:
            'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)',
        };
      case 'triangle':
        return {...base, clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)'};
      case 'teardrop':
        return {...base, borderRadius: '50% 50% 50% 15%'};
      case 'bean':
        return {...base, borderRadius: '60% 40% 60% 40% / 60% 40% 60% 40%'};
      default:
        return {...base, borderRadius: '50%'};
    }
  };

  const getThumbStyle = () => {
    const shapeStyle = getThumbShapeStyle(thumbShape, thumbSize);
    const isClipPathShape = ['diamond', 'hexagon', 'star', 'triangle'].includes(
      thumbShape,
    );

    return {
      position: 'absolute',
      ...shapeStyle,
      top: `${padding}px`,
      left: `${padding}px`,
      background: thumbColor,
      transform: `translateX(${translate}px)`,
      transition: isDragging
        ? 'none'
        : `transform ${params.duration}ms ${easing}`,
      cursor: 'pointer',
      ...(visualStyle === 'shadow'
        ? {boxShadow: '0 2px 8px rgba(0,0,0,0.15)'}
        : {}),
      ...(visualStyle === 'glow'
        ? {
            boxShadow: isOn
              ? `0 0 20px ${activeColor}60`
              : '0 2px 4px rgba(0,0,0,0.1)',
          }
        : {}),
      ...(visualStyle === 'outline' && !isClipPathShape
        ? {
            // For regular shapes, use normal border
            border: `2px solid ${isOn ? activeColor : '#d1d5db'}`,
            background: 'white',
          }
        : {}),
      ...(visualStyle === 'outline' && isClipPathShape
        ? {
            // For clipPath shapes, set white background
            background: 'white',
          }
        : {}),
      ...(visualStyle === 'elevated'
        ? {
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            transform: `translateX(${translate}px) translateY(-1px)`,
          }
        : {}),
    };
  };

  // Create border element for clipPath shapes with outline style
  const getThumbBorderElement = () => {
    const isClipPathShape = ['diamond', 'hexagon', 'star', 'triangle'].includes(
      thumbShape,
    );

    if (visualStyle !== 'outline' || !isClipPathShape) return null;

    const borderSize = thumbSize + 4; // 2px border on each side
    const borderOffset = -2; // offset to center the border
    const shapeStyle = getThumbShapeStyle(thumbShape, borderSize);

    return (
      <div
        style={{
          position: 'absolute',
          ...shapeStyle,
          top: `${padding + borderOffset}px`,
          left: `${padding + borderOffset}px`,
          background: isOn ? activeColor : '#d1d5db',
          transform: `translateX(${translate}px)`,
          transition: isDragging
            ? 'none'
            : `transform ${params.duration}ms ${easing}`,
          zIndex: 1,
        }}
      />
    );
  };

  return (
    <div className="flex items-center justify-center">
      <div ref={toggleRef} onMouseEnter={handleHover} style={getTrackStyle()}>
        {getThumbBorderElement()}
        <div style={{...getThumbStyle(), zIndex: 2, position: 'relative'}} />
      </div>
    </div>
  );
}
