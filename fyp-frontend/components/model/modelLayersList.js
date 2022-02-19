import React from "react";
import {
  ListSubheader,
  List,
  ListItemButton,
  ListItemText,
  Collapse,
} from "@mui/material/";

const ModelLayersList = ({ listOptionsInit }) => {
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
          Layer types and the available options
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
                  <span class="material-icons">chevron_right</span>
                ) : (
                  <span class="material-icons">chevron_left</span>
                )}
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
          </ListItemButton>
        </>
      ))}
    </List>
  );
};

export default ModelLayersList;
