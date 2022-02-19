import React from "react";
import {
  ListSubheader,
  List,
  ListItemButton,
  ListItemText,
  Collapse,
} from "@mui/material/";

const OptionsDynamicList = ({ listOptionsInit, title }) => {
  const [listValues, setListValues] = React.useState(listOptionsInit);

  const handleClick = (listItemName) => {
    setListValues({
      ...listValues,
      collapsables: {
        ...listValues["collapsables"],
        [listItemName]: !listValues["collapsables"][listItemName],
      },
    });
  };
  return (
    <List
      sx={{ width: "100%", bgcolor: "background.paper" }}
      aria-labelledby="nested-list-subheader"
      subheader={
        <ListSubheader component="div" id="nested-list-subheader">
          {title}
        </ListSubheader>
      }
    >
      {listValues.listOptions.map((option) => (
        <>
          <ListItemButton onClick={(event) => handleClick(option.name)}>
            <ListItemText primary={option.name} />
            {listValues.collapsables.hasOwnProperty(option.name) && (
              <>
                {!listValues.collapsables[option.name] ? (
                  <span className="material-icons">expand_more</span>
                ) : (
                  <span className="material-icons">expand_less</span>
                )}
              </>
            )}
          </ListItemButton>
          {listValues.collapsables.hasOwnProperty(option.name) && (
            <>
              <Collapse
                in={listValues.collapsables[option.name]}
                timeout="auto"
                unmountOnExit
              >
                <List component="div" disablePadding>
                  {option.values.map((value) => (
                    <ListItemButton sx={{ pl: 4 }}>
                      <ListItemText primary={value} />
                    </ListItemButton>
                  ))}
                </List>
              </Collapse>
            </>
          )}
        </>
      ))}
    </List>
  );
};

export default OptionsDynamicList;
